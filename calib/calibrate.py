"""Camera calibration from a video or a folder of photos (adapted from calib/calibration.py).

A video is streamed and only the selected frames are written to disk. With --fotos
the photos stay where they are and steps 1 and 2 change: every photo is read, and
the blur percentile and the motion check do not apply, because capture_stills.py
already checked each photo and consecutive photos are not consecutive frames.

With --fotos the target is not assumed flat. A printed sheet on a board bends by a
few mm, which biases K, so the calibration also estimates the real 3D position of
every board corner (cv2.calibrateCameraRO) and triangulates the marker corners from
all views. Both go to calib/alvo_medido.json. Use --alvo-plano to skip this.

  1. Scan every `--stride`-th frame and detect the board with alvo/referencia.py.
  2. Reject frames with a partial board, with low Laplacian variance inside the
     board region, or with a moving board. Motion is measured separately because
     rolling shutter shears the board without lowering the Laplacian variance.
  3. Run a provisional calibration on an evenly spaced sample, only to get each
     candidate's pose (tilt depends on K).
  4. Greedy farthest-point selection over image position, tilt direction and
     distance, skipping near-duplicates of a chosen frame: corners that barely
     moved and a board normal that barely turned. Corner motion alone is not
     enough, since a small board tilted by 25 degrees moves its corners only ~10 px.
  5. Calibrate with the radial k1, k2 model on the selected frames.
  6. Measure stability: bootstrap for a video; for photos, 5-fold validation, which
     also gives the error on photos left out of the fit.

Run from anywhere: .venv/bin/python calib/calibrate.py [--video FILE | --fotos FOLDER]
Deterministic: same input and arguments give the same outputs.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alvo.referencia import cantos_marcador, cantos_tabuleiro, carregar_geometria  # noqa: E402

RADIAL_K1K2 = cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST
TILT_WARN_DEG = 30.0
RMS_WARN_PX = 1.0
PROVISIONAL_SAMPLE = 25
MIN_NORMAL_CHANGE = float(np.sin(np.radians(5)))   # below this, the board orientation is the same
PHOTO_SUFFIXES = {".png", ".jpg", ".jpeg"}
VIDEO_BLUR_PERCENTILE = 30
VIDEO_MAX_FRAMES = 40
VIDEO_MIN_MOVE_PX = 15.0
PHOTO_MIN_MOVE_PX = 2.0   # each photo was taken on purpose: only true duplicates go
BOOTSTRAP_RUNS = 100
BOOTSTRAP_SEED = 0
F_STABILITY_WARN = 0.02       # bootstrap sd of fx or fy, as a fraction of the value
PP_STABILITY_WARN_PX = 20.0   # bootstrap sd of cx or cy
COVERAGE_WARN = 0.75          # fraction of 4x4 image cells reached by some corner
COVERAGE_GRID = 4             # same grid as the live view of capture_stills.py
PP_OFFSET_WARN = 0.05         # principal point off-centre, as a fraction of the image size
ASPECT_WARN = 0.01            # relative difference between fx and fy
SIDE_TILT_DEG = 15.0          # a photo counts for a tilt direction above this tilt
SIDE_MIN_PHOTOS = 3
SIDES = ("esquerda", "direita", "topo", "base")
FOLDS = 5
MIN_MARKER_VIEWS = 10


def board_sharpness(gray, corners):
    """Laplacian variance restricted to the convex hull of the board corners."""
    pts = corners.reshape(-1, 2)
    x0, y0 = np.floor(pts.min(0)).astype(int)
    x1, y1 = np.ceil(pts.max(0)).astype(int) + 1
    crop = gray[y0:y1, x0:x1]
    mask = np.zeros(crop.shape, np.uint8)
    hull = cv2.convexHull((pts - [x0, y0]).astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 1)
    return float(cv2.Laplacian(crop, cv2.CV_64F)[mask > 0].var())


def scan_video(path, geo, stride, margin):
    """Stream the video and return board candidates plus scan counters."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {path}")
    candidates, size = [], None
    stats = {"decoded": 0, "scanned": 0, "no_board": 0, "at_border": 0}
    idx = -1
    while cap.grab():
        idx += 1
        stats["decoded"] += 1
        if idx % stride:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            continue
        stats["scanned"] += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        corners = cantos_tabuleiro(gray, geo)
        if corners is None:
            stats["no_board"] += 1
            continue
        pts = corners.reshape(-1, 2)
        if (pts.min() < margin or pts[:, 0].max() > size[0] - 1 - margin
                or pts[:, 1].max() > size[1] - 1 - margin):
            stats["at_border"] += 1
            continue
        candidates.append({"frame": idx, "name": str(idx), "time_s": cap.get(cv2.CAP_PROP_POS_MSEC) / 1000,
                           "corners": corners, "sharpness": board_sharpness(gray, corners)})
        if stats["scanned"] % 100 == 0:
            print(f"  scanned {stats['scanned']} frames, {len(candidates)} with the board", flush=True)
    cap.release()
    return candidates, size, stats


def scan_photos(folder, geo, margin):
    """Read every photo in a folder and return board candidates plus counters."""
    paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in PHOTO_SUFFIXES)
    if not paths:
        raise SystemExit(f"No photos found in {folder}")
    candidates, size = [], None
    stats = {"photos": len(paths), "no_board": 0, "at_border": 0}
    for idx, path in enumerate(paths):
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise SystemExit(f"Could not read {path}")
        if size is None:
            size = gray.shape[::-1]
        elif gray.shape[::-1] != size:
            raise SystemExit(f"{path.name} is {gray.shape[::-1]}, expected {size}: one K needs one camera mode.")
        corners = cantos_tabuleiro(gray, geo)
        if corners is None:
            stats["no_board"] += 1
            continue
        pts = corners.reshape(-1, 2)
        if (pts.min() < margin or pts[:, 0].max() > size[0] - 1 - margin
                or pts[:, 1].max() > size[1] - 1 - margin):
            stats["at_border"] += 1
            continue
        candidates.append({"frame": idx, "name": path.name, "path": path, "time_s": None,
                           "corners": corners, "marker": cantos_marcador(gray, geo),
                           "sharpness": board_sharpness(gray, corners)})
    return candidates, size, stats


def add_speed(frames, stride):
    """Mean corner displacement (px/frame) to the adjacent scanned frames, worst side."""
    by_index = {f["frame"]: f for f in frames}
    for f in frames:
        steps = []
        for other in (f["frame"] - stride, f["frame"] + stride):
            if other in by_index:
                shift = np.linalg.norm(by_index[other]["corners"] - f["corners"], axis=2).mean()
                steps.append(float(shift) / stride)
        f["speed"] = max(steps) if steps else np.inf


def bootstrap_intrinsics(objp, frames, size):
    """Standard deviation of fx, fy, cx, cy when resampling frames with replacement."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = []
    for _ in range(BOOTSTRAP_RUNS):
        sample = [frames[i] for i in rng.integers(0, len(frames), len(frames))]
        K = calibrate(objp, sample, size)[1]
        values.append([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
    return np.std(values, axis=0)


def calibrate(objp, frames, size):
    obj = [objp] * len(frames)
    img = [f["corners"] for f in frames]
    return cv2.calibrateCameraExtended(obj, img, size, None, None, flags=RADIAL_K1K2)


def calibrate_refined(objp, frames, size, fixed_point):
    """Calibrate while estimating the real 3D board corners. Same return order as
    `calibrate`, with the corners appended."""
    result = cv2.calibrateCameraROExtended([objp] * len(frames), [f["corners"] for f in frames], size,
                                           fixed_point, None, None, flags=RADIAL_K1K2)
    rms, K, dist, rvecs, tvecs, shape, std_in = result[:7]
    return rms, K, dist, rvecs, tvecs, std_in, shape.reshape(-1, 3).astype(np.float32)


def reprojection_rms(frames, points, K, dist, key="corners"):
    """RMS error of a known 3D shape, with the pose solved per frame from the board."""
    sq = []
    for f in frames:
        proj, _ = cv2.projectPoints(points, f["rvec"], f["tvec"], K, dist)
        sq.append(((f[key].reshape(-1, 2) - proj.reshape(-1, 2)) ** 2).sum(1))
    return float(np.sqrt(np.concatenate(sq).mean()))


def solve_board_poses(frames, shape, K, dist):
    for f in frames:
        _, f["rvec"], f["tvec"] = cv2.solvePnP(shape, f["corners"], K, dist)


def measure_marker(frames, K, dist):
    """Least-squares intersection of the rays to each marker corner, in the board frame."""
    views = [f for f in frames if f.get("marker") is not None]
    if len(views) < MIN_MARKER_VIEWS:
        return None
    A, b = np.zeros((4, 3, 3)), np.zeros((4, 3))
    for f in views:
        R = cv2.Rodrigues(f["rvec"])[0]
        origin = -R.T @ f["tvec"].ravel()
        rays = cv2.undistortPoints(f["marker"], K, dist).reshape(-1, 2)
        for i, (x, y) in enumerate(rays):
            direction = R.T @ np.array([x, y, 1.0])
            direction /= np.linalg.norm(direction)
            P = np.eye(3) - np.outer(direction, direction)
            A[i] += P
            b[i] += P @ origin
    return np.array([np.linalg.solve(A[i], b[i]) for i in range(4)], np.float32)


def cross_validate(objp, frames, size, fixed_point, refine, nominal_marker):
    """Refit on 4/5 of the photos and score the other 1/5, which the fit never saw."""
    order = np.random.default_rng(BOOTSTRAP_SEED).permutation(len(frames))
    params, held_out, marker_nominal, marker_measured = [], [], [], []
    for k in range(FOLDS):
        test_ids = set(order[k::FOLDS].tolist())
        test = [f for i, f in enumerate(frames) if i in test_ids]
        train = [f for i, f in enumerate(frames) if i not in test_ids]
        if refine:
            _, K, dist, *_, shape = calibrate_refined(objp, train, size, fixed_point)
        else:
            _, K, dist, *_ = calibrate(objp, train, size)
            shape = objp
        params.append([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
        solve_board_poses(train + test, shape, K, dist)
        held_out.append(reprojection_rms(test, shape, K, dist))
        marker = measure_marker(train, K, dist)
        seen = [f for f in test if f.get("marker") is not None]
        if marker is not None and seen:
            marker_nominal.append(reprojection_rms(seen, nominal_marker, K, dist, "marker"))
            marker_measured.append(reprojection_rms(seen, marker, K, dist, "marker"))
    return {"sd": np.std(params, axis=0), "held_out": held_out,
            "marker_nominal": marker_nominal, "marker_measured": marker_measured}


def tilt_side(normal):
    """Which side of the board is closest to the camera (image axes: x right, y down)."""
    if abs(normal[0]) >= abs(normal[1]):
        return "direita" if normal[0] > 0 else "esquerda"
    return "base" if normal[1] > 0 else "topo"


def tilt_and_normal(rvec):
    """Angle between the board normal and the optical axis, and the unit normal."""
    n = cv2.Rodrigues(rvec)[0][:, 2]
    n = n * np.sign(n[2])
    return float(np.degrees(np.arccos(np.clip(n[2], -1, 1)))), n


def add_pose_features(frames, objp, K, dist, size):
    for f in frames:
        ok, rvec, tvec = cv2.solvePnP(objp, f["corners"], K, dist)
        if not ok:
            f["feature"] = None
            continue
        tilt, n = tilt_and_normal(rvec)
        center = f["corners"].reshape(-1, 2).mean(0)
        f["tilt_deg"] = tilt
        f["feature"] = np.array([center[0] / size[0], center[1] / size[1], n[0], n[1],
                                 np.log(np.linalg.norm(tvec))])


def select_frames(frames, max_frames, min_move_px):
    """Greedy farthest-point sampling, seeded with the most tilted frame."""
    pool = [f for f in frames if f["feature"] is not None]
    if not pool:
        return []
    feats = np.array([f["feature"] for f in pool])
    pts = np.array([f["corners"].reshape(-1, 2) for f in pool])
    alive = np.ones(len(pool), bool)
    nearest = np.full(len(pool), np.inf)
    pick = int(np.argmax([f["tilt_deg"] for f in pool]))
    chosen = []
    while True:
        chosen.append(pick)
        alive[pick] = False
        move = np.linalg.norm(pts - pts[pick], axis=2).mean(axis=1)
        turn = np.linalg.norm(feats[:, 2:4] - feats[pick, 2:4], axis=1)
        alive &= (move >= min_move_px) | (turn >= MIN_NORMAL_CHANGE)
        nearest = np.minimum(nearest, np.linalg.norm(feats - feats[pick], axis=1))
        if len(chosen) == max_frames or not alive.any():
            break
        pick = int(np.argmax(np.where(alive, nearest, -np.inf)))
    return sorted((pool[i] for i in chosen), key=lambda f: f["frame"])


def clear_outputs(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in list(out_dir.glob("frame_*.png")) + list(out_dir.glob("mosaic.jpg")):
        old.unlink()


def thumbnail(frame, f, pattern):
    vis = cv2.drawChessboardCorners(frame.copy(), pattern, f["corners"], True)
    vis = cv2.resize(vis, (320, 320 * frame.shape[0] // frame.shape[1]))
    cv2.putText(vis, f"{f['name']} tilt {f['tilt_deg']:.0f}", (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return vis


def write_mosaic(tiles, out_dir):
    cols = 6
    tiles = tiles + [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(str(out_dir / "mosaic.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def save_frames(path, chosen, out_dir, pattern):
    """Second streaming pass: write the chosen frames and a contact sheet."""
    clear_outputs(out_dir)
    wanted = {f["frame"]: f for f in chosen}
    thumbs = {}
    cap = cv2.VideoCapture(str(path))
    idx = -1
    while wanted.keys() - thumbs.keys() and cap.grab():
        idx += 1
        if idx not in wanted:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            raise SystemExit(f"Could not re-read frame {idx}")
        cv2.imwrite(str(out_dir / f"frame_{idx:05d}.png"), frame)
        thumbs[idx] = thumbnail(frame, wanted[idx], pattern)
    cap.release()
    write_mosaic([thumbs[k] for k in sorted(thumbs)], out_dir)


def save_photo_mosaic(chosen, out_dir, pattern):
    """The photos are already on disk, so only the contact sheet is written."""
    clear_outputs(out_dir)
    write_mosaic([thumbnail(cv2.imread(str(f["path"])), f, pattern) for f in chosen], out_dir)


def coverage_grid(frames, size, cols=8, rows=6):
    """Number of chosen frames with at least one corner in each image cell."""
    grid = np.zeros((rows, cols), int)
    for f in frames:
        pts = f["corners"].reshape(-1, 2)
        cells = {(min(int(y * rows / size[1]), rows - 1), min(int(x * cols / size[0]), cols - 1))
                 for x, y in pts}
        for cell in cells:
            grid[cell] += 1
    return grid


def write_target(path, source, shape, marker):
    data = {
        "fonte": source,
        "unidade": "mm",
        "referencial": "o mesmo de alvo/geometria_alvo.json; z positivo para dentro da folha",
        "cantos_tabuleiro_mm": np.round(shape, 4).tolist(),
        "cantos_marcador_mm": None if marker is None else np.round(marker, 4).tolist(),
    }
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")


def write_yaml(path, source, size, rms, K, dist, n_frames, max_tilt, warnings):
    data = {
        "source": source,
        "image_width": int(size[0]),
        "image_height": int(size[1]),
        "model": "radial k1,k2 (p1=p2=k3=0)",
        "rms_px": float(rms),
        "n_frames": int(n_frames),
        "max_tilt_deg": float(max_tilt),
        "K": K.tolist(),
        "dist_k1_k2_p1_p2_k3": dist.ravel().tolist(),
        "quality_warnings": warnings,
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def write_report(path, a, source, size, stats, rejected_blur, rejected_moving, blur_thr, n_cand, rows,
                 rms, K, dist, std, boot, grid, warnings, sides, validation, target):
    photos = a.fotos is not None
    unit = "Foto" if photos else "Quadro"
    k1, k2 = dist.ravel()[:2]
    tilts = np.array([r["tilt_deg"] for r in rows])
    errs = np.array([r["error_px"] for r in rows])
    hist, edges = np.histogram(tilts, bins=[0, 15, 30, 45, 60, 90])
    L = ["# Relatório de calibração", ""]
    L += ["## Avisos", ""]
    L += [f"- **AVISO:** {w}" for w in warnings] or ["Nenhum aviso: inclinação máxima e RMS dentro dos limites."]
    L += ["", "## Resumo", "",
          f"- {'Fotos' if photos else 'Vídeo'}: `{source}`",
          f"- Resolução: {size[0]} x {size[1]} px",
          "- Modelo: radial k1, k2 (p1 = p2 = k3 = 0)",
          f"- {unit}s usad{'a' if photos else 'o'}s: {len(rows)}",
          f"- RMS de reprojeção: {rms:.4f} px",
          f"- Inclinação: mínima {tilts.min():.1f}°, mediana {np.median(tilts):.1f}°, máxima {tilts.max():.1f}°",
          "", "## Intrínsecos", "",
          (f"Estabilidade: desvio entre os {FOLDS} ajustes da validação cruzada. " if photos else
           f"Estabilidade: desvio em {BOOTSTRAP_RUNS} reamostragens dos quadros selecionados. ") +
          "É a medida a considerar; o desvio do OpenCV supõe um modelo correto e costuma ser menor.",
          "",
          "| Parâmetro | Valor | Desvio OpenCV | Estabilidade |", "|---|---|---|---|",
          f"| fx (px) | {K[0, 0]:.3f} | {std[0]:.3f} | {boot[0]:.3f} |",
          f"| fy (px) | {K[1, 1]:.3f} | {std[1]:.3f} | {boot[1]:.3f} |",
          f"| cx (px) | {K[0, 2]:.3f} | {std[2]:.3f} | {boot[2]:.3f} |",
          f"| cy (px) | {K[1, 2]:.3f} | {std[3]:.3f} | {boot[3]:.3f} |",
          f"| k1 | {k1:.6f} | {std[4]:.6f} | |",
          f"| k2 | {k2:.6f} | {std[5]:.6f} | |",
          "", "K =", "", "```"]
    L += ["  ".join(f"{v:12.4f}" for v in line) for line in K]
    L += ["```", "", "Distorção (k1, k2, p1, p2, k3) = " + ", ".join(f"{v:.6f}" for v in dist.ravel()),
          "", "## Seleção", "",
          f"| Etapa | {unit}s |", "|---|---|"]
    if photos:
        L += [f"| Na pasta | {stats['photos']} |"]
    else:
        L += [f"| Decodificados | {stats['decoded']} |",
              f"| Analisados (1 a cada {a.stride}) | {stats['scanned']} |"]
    L += [f"| Sem o tabuleiro inteiro | {stats['no_board']} |",
          f"| Tabuleiro a menos de {a.margin} px da borda | {stats['at_border']} |"]
    if not photos:
        L += [f"| Borrados (Laplaciano < {blur_thr:.1f}, percentil {a.blur_percentile:g}) | {rejected_blur} |",
              f"| Em movimento (acima de {a.max_speed:g} px/quadro) | {rejected_moving} |"]
    L += [f"| Candidatos | {n_cand} |",
          f"| Selecionados (até {a.max_frames}, sem quase duplicatas) | {len(rows)} |"]
    if validation:
        held = validation["held_out"]
        L += ["", "## Validação em fotos fora do ajuste", "",
              f"Erro de reprojeção do tabuleiro: {min(held):.3f} a {max(held):.3f} px "
              f"nos {FOLDS} grupos (RMS do ajuste: {rms:.3f} px)."]
        if validation["marker_measured"]:
            L += [f"Erro do marcador previsto pela pose do tabuleiro: "
                  f"{np.mean(validation['marker_nominal']):.2f} px com a posição nominal, "
                  f"{np.mean(validation['marker_measured']):.2f} px com a posição medida."]
    if target:
        shape, marker, objm, nx = target
        L += ["", "## Forma medida do alvo", "",
              f"Salva em `{a.alvo_medido.name}`. Desvio da folha em relação ao plano nominal (z, mm), "
              "uma linha por fileira de cantos do tabuleiro:", "", "```"]
        L += [" ".join(f"{v:6.2f}" for v in line) for line in shape[:, 2].reshape(-1, nx)]
        L += ["```"]
        if marker is not None:
            shift = np.linalg.norm(marker - objm, axis=1)
            edges_mm = np.linalg.norm(marker - np.roll(marker, -1, axis=0), axis=1)
            L += ["", f"Marcador: deslocamento de {shift.min():.2f} a {shift.max():.2f} mm da posição nominal; "
                  f"lados medidos de {edges_mm.min():.2f} a {edges_mm.max():.2f} mm."]
    L += ["", "## Cobertura", "",
          "Quadros por faixa de inclinação:", "",
          "| Faixa | Quadros |", "|---|---|"]
    L += [f"| {edges[i]:.0f}° a {edges[i + 1]:.0f}° | {hist[i]} |" for i in range(len(hist))]
    L += ["", f"Lado do alvo mais perto da câmera, com inclinação acima de {SIDE_TILT_DEG:.0f}°: " +
          ", ".join(f"{name} {sides[name]}" for name in SIDES) + ".",
          "", "Quadros com algum canto em cada célula da imagem (grade 8 x 6):", "", "```"]
    L += [" ".join(f"{v:3d}" for v in line) for line in grid]
    L += ["```", "", "## Por quadro", "",
          f"Erro = RMS de reprojeção de cada {unit.lower()}. Média dos erros: {errs.mean():.3f} px; máximo: {errs.max():.3f} px.",
          "Inclinação = ângulo entre a normal do alvo e o eixo óptico.", "",
          f"| {unit} | Tempo (s) | Erro (px) | Inclinação (°) | Distância (mm) | Laplaciano |",
          "|---|---|---|---|---|---|"]
    L += [f"| {r['name']} | {'' if r['time_s'] is None else format(r['time_s'], '.2f')} | "
          f"{r['error_px']:.3f} | {r['tilt_deg']:.1f} | {r['distance_mm']:.0f} | {r['sharpness']:.1f} |"
          for r in rows]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = ap.add_mutually_exclusive_group()
    source.add_argument("--video", type=Path, default=ROOT / "data/raw/E0_calibracao")
    source.add_argument("--fotos", type=Path, help="folder of photos, used instead of the video")
    ap.add_argument("--frames-dir", type=Path, default=ROOT / "data/proc/calib_frames")
    ap.add_argument("--yaml", type=Path, default=ROOT / "calib/camera.yaml")
    ap.add_argument("--report", type=Path, default=ROOT / "calib/relatorio.md")
    ap.add_argument("--stride", type=int, default=1, help="analyse one frame out of N")
    ap.add_argument("--margin", type=int, default=8, help="min distance (px) from corners to image edge")
    ap.add_argument("--blur-percentile", type=float, default=None,
                    help="reject candidates below this percentile of board Laplacian variance "
                         f"(default: {VIDEO_BLUR_PERCENTILE} for a video, 0 for photos)")
    ap.add_argument("--max-speed", type=float, default=1.0,
                    help="video only: reject candidates whose corners move faster than this (px/frame)")
    ap.add_argument("--max-frames", type=int, default=None, help="default: 40 for a video, all photos")
    ap.add_argument("--alvo-plano", action="store_true", help="photos only: keep the nominal flat target")
    ap.add_argument("--alvo-medido", type=Path, default=ROOT / "calib/alvo_medido.json")
    ap.add_argument("--min-move", type=float, default=None,
                    help="a frame is a near-duplicate below this mean corner displacement (px) if its "
                         f"board normal also turned less than 5 degrees (default: {VIDEO_MIN_MOVE_PX:g} "
                         f"for a video, {PHOTO_MIN_MOVE_PX:g} for photos)")
    a = ap.parse_args()

    geo = carregar_geometria()
    objp = geo["objp_tabuleiro"]

    photos = a.fotos is not None
    if a.blur_percentile is None:
        a.blur_percentile = 0 if photos else VIDEO_BLUR_PERCENTILE
    if a.min_move is None:
        a.min_move = PHOTO_MIN_MOVE_PX if photos else VIDEO_MIN_MOVE_PX
    refine = photos and not a.alvo_plano
    source_path = a.fotos if photos else a.video
    print(f"Scanning {source_path} ...", flush=True)
    if photos:
        found, size, stats = scan_photos(a.fotos, geo, a.margin)
    else:
        found, size, stats = scan_video(a.video, geo, a.stride, a.margin)
    if len(found) < 10:
        raise SystemExit(f"Only {len(found)} frames with the full board: {stats}")

    # Multithreaded calibrateCamera sums in a varying order, so K changes between runs.
    cv2.setNumThreads(1)

    blur_thr = float(np.percentile([f["sharpness"] for f in found], a.blur_percentile))
    sharp = [f for f in found if f["sharpness"] >= blur_thr]
    rejected_blur = len(found) - len(sharp)
    if photos:
        still = sharp
    else:
        add_speed(found, a.stride)
        still = [f for f in sharp if f["speed"] <= a.max_speed]
    rejected_moving = len(sharp) - len(still)
    if len(still) < 10:
        raise SystemExit(f"Only {len(still)} sharp frames below {a.max_speed} px/frame.")

    sample = [still[i] for i in np.linspace(0, len(still) - 1, min(PROVISIONAL_SAMPLE, len(still))).astype(int)]
    _, K0, dist0, *_ = calibrate(objp, sample, size)
    add_pose_features(still, objp, K0, dist0, size)
    if a.max_frames is None:
        a.max_frames = len(still) if photos else VIDEO_MAX_FRAMES
    chosen = select_frames(still, a.max_frames, a.min_move)
    if len(chosen) < 10:
        raise SystemExit(f"Only {len(chosen)} frames selected; lower --min-move or record more poses.")

    fixed_point = geo["padrao"][0] - 1
    if refine:
        rms, K, dist, rvecs, tvecs, std_in, shape = calibrate_refined(objp, chosen, size, fixed_point)
    else:
        rms, K, dist, rvecs, tvecs, std_in, _, _ = calibrate(objp, chosen, size)
        shape = objp
    std = std_in.ravel()[:6]
    validation = None
    if photos:
        print("Cross-validating ...", flush=True)
        validation = cross_validate(objp, chosen, size, fixed_point, refine, geo["objp_marcador"])
        boot = validation["sd"]
    else:
        boot = bootstrap_intrinsics(objp, chosen, size)
    grid = coverage_grid(chosen, size)
    coverage = float((coverage_grid(chosen, size, COVERAGE_GRID, COVERAGE_GRID) > 0).mean())

    rows = []
    sides = dict.fromkeys(SIDES, 0)
    for f, r, t in zip(chosen, rvecs, tvecs):
        proj, _ = cv2.projectPoints(shape, r, t, K, dist)
        diff = f["corners"].reshape(-1, 2) - proj.reshape(-1, 2)
        f["rvec"], f["tvec"] = r, t
        f["tilt_deg"], normal = tilt_and_normal(r)
        if f["tilt_deg"] > SIDE_TILT_DEG:
            sides[tilt_side(normal)] += 1
        rows.append({"name": f["name"], "time_s": f["time_s"], "sharpness": f["sharpness"],
                     "error_px": float(np.sqrt((diff ** 2).sum(1).mean())),
                     "tilt_deg": f["tilt_deg"], "distance_mm": float(np.linalg.norm(t))})

    max_tilt = max(r["tilt_deg"] for r in rows)
    warnings = []
    if max_tilt < TILT_WARN_DEG:
        warnings.append(f"inclinação máxima de {max_tilt:.1f}°, abaixo de {TILT_WARN_DEG:.0f}°.")
    if rms > RMS_WARN_PX:
        warnings.append(f"RMS de {rms:.3f} px, acima de {RMS_WARN_PX:.0f} px.")
    if max(boot[0] / K[0, 0], boot[1] / K[1, 1]) > F_STABILITY_WARN:
        warnings.append(f"distância focal instável: desvio de {boot[0]:.1f} px em fx e "
                        f"{boot[1]:.1f} px em fy, acima de {F_STABILITY_WARN:.0%}.")
    if max(boot[2], boot[3]) > PP_STABILITY_WARN_PX:
        warnings.append(f"ponto principal instável: desvio de {boot[2]:.1f} px em cx e "
                        f"{boot[3]:.1f} px em cy, acima de {PP_STABILITY_WARN_PX:.0f} px.")
    if coverage < COVERAGE_WARN:
        warnings.append(f"cobertura baixa: cantos em {coverage:.0%} das células da grade 4x4, "
                        f"abaixo de {COVERAGE_WARN:.0%}.")
    if abs(K[0, 2] / size[0] - 0.5) > PP_OFFSET_WARN or abs(K[1, 2] / size[1] - 0.5) > PP_OFFSET_WARN:
        warnings.append(f"ponto principal ({K[0, 2]:.0f}, {K[1, 2]:.0f}) longe do centro da imagem "
                        f"({size[0] / 2:.0f}, {size[1] / 2:.0f}).")
    if abs(K[0, 0] / K[1, 1] - 1) > ASPECT_WARN:
        warnings.append(f"fx e fy diferem em {abs(K[0, 0] / K[1, 1] - 1):.1%}, acima de {ASPECT_WARN:.0%}.")
    missing = [name for name in SIDES if sides[name] < SIDE_MIN_PHOTOS]
    if missing:
        warnings.append("poucas fotos com o alvo inclinado com este lado mais perto da câmera: "
                        + ", ".join(missing) + ".")

    if photos:
        save_photo_mosaic(chosen, a.frames_dir, geo["padrao"])
    else:
        save_frames(a.video, chosen, a.frames_dir, geo["padrao"])
    source_rel = Path(os.path.relpath(source_path.resolve(), ROOT)).as_posix()
    target = None
    if refine:
        marker = measure_marker(chosen, K, dist)
        write_target(a.alvo_medido, source_rel, shape, marker)
        target = (shape, marker, geo["objp_marcador"], geo["padrao"][0])
    write_yaml(a.yaml, source_rel, size, rms, K, dist, len(chosen), max_tilt, warnings)
    write_report(a.report, a, source_rel, size, stats, rejected_blur, rejected_moving, blur_thr, len(still),
                 rows, rms, K, dist, std, boot, grid, warnings, sides, validation, target)

    print(f"Scan: {stats}, blurred: {rejected_blur} (threshold {blur_thr:.1f}), "
          f"moving: {rejected_moving}, selected: {len(chosen)}")
    print(f"Stability sd fx, fy, cx, cy: {boot.round(2)}")
    if validation:
        print(f"Held-out error: {min(validation['held_out']):.3f} to {max(validation['held_out']):.3f} px")
    print(f"RMS {rms:.4f} px, max tilt {max_tilt:.1f} deg")
    print(f"K =\n{K}\ndist = {dist.ravel()}")
    for w in warnings:
        print(f"WARNING: {w}")


if __name__ == "__main__":
    main()
