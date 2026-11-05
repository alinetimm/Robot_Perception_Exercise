"""Pose through occlusions: hold, Lucas-Kanade on the corners, and Lucas-Kanade on the ring.

Synthetic occlusions paint a grey cover over the marker in E2 and E3, for windows of
N frames; the board stays visible, so the reference exists during the occlusion. The
real occlusions of E5 are the stretches where the hand hides the marker.

The covariance after k frames without detection is Sigma_k = Sigma + k Q. Q is fitted
for each method on the even windows of E2 only. Odd windows of E2, E3 and E5 are
independent checks. Sigma uses the speed model of results/nees.json.

Reads data/raw/, data/proc/frames/ and results/nees.json. Writes results/occlusion.json,
and the per-frame values to data/proc/occlusion/.

Run: .venv/bin/python scripts/04_occlusion.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alvo.referencia import carregar_geometria_medida  # noqa: E402
from src.covariance import (CHI2_6_HIGH, CHI2_6_LOW, jacobian, nees, pose_covariance, project,  # noqa: E402
                            sigma_at_speed, transfer_to_marker)
from src.frames import corners, load, pose  # noqa: E402
from src.geometry import pose_error  # noqa: E402
from src.occlusion import (CornerTracker, RingTracker, fit_growth, jumped, paint_cover, pose_near,  # noqa: E402
                           ring_mask, speed_limit, trusted_anchor)
from src.pose import expected_marker_pose, load_camera, marker_model  # noqa: E402

RAW = ROOT / "data/raw"
OUT = ROOT / "data/proc/occlusion"
WINDOWS = (15, 30, 60)
GAP_FRAMES = 45
FIRST_FRAME = 60
MAX_REAL_FRAMES = 150
MIN_REAL_FRAMES = 5
METHODS = ("hold", "corners", "ring")
K_BINS = [0, 5, 15, 30, 60, 1000]


def read_gray(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            return frames
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))


def reference_at(row):
    R, t = pose(row, "ref")
    return type("Pose", (), {"R": R, "t": t})


def true_corners(row, model, camera, fallback):
    """Where the marker is, for painting: from the board, else from the detector."""
    if row["board_ok"]:
        R, t = expected_marker_pose(reference_at(row), model)
        return project(model.points, R, t, camera).reshape(-1, 1, 2).astype(np.float32)
    if row["marker_ok"]:
        return np.array([[row[f"mc{i}_x"], row[f"mc{i}_y"]] for i in range(4)], np.float32).reshape(-1, 1, 2)
    return fallback


def measure(estimate, speed, row, board_points, model, camera, speed_model):
    """Error against the reference, with the covariance before the growth term."""
    reference = reference_at(row)
    R_exp, t_exp = expected_marker_pose(reference, model)
    sigma = float(sigma_at_speed(speed, speed_model["base_px"], speed_model["gain"]))
    cov = sigma ** 2 * (pose_covariance(jacobian(model.points, estimate.R, estimate.t, camera), 1.0)
                        + speed_model["board_ratio"] ** 2 * transfer_to_marker(
                            pose_covariance(jacobian(board_points, reference.R, reference.t, camera), 1.0),
                            reference.R, model.t_target))
    return pose_error(estimate.R, estimate.t, R_exp, t_exp), cov


def run_window(frames, table, start, length, painted, context):
    """Follow one occlusion; returns one record per method and frame."""
    geo_points, model, camera, speed_model = context
    anchor = trusted_anchor(table.iloc[max(0, start - 12):start].to_dict("index"), start)
    last = table.loc[anchor]
    R0, t0 = pose(last, "m")
    first = np.array([[last[f"mc{i}_x"], last[f"mc{i}_y"]] for i in range(4)], np.float32).reshape(-1, 1, 2)
    before = corners(table.iloc[max(0, anchor - 10):anchor + 1], "mc", 4)
    steps = np.linalg.norm(np.diff(before, axis=0), axis=2).mean(axis=1)
    limit = speed_limit(np.nanmedian(steps) if np.isfinite(steps).any() else 0.0)
    image = frames[anchor]
    trackers = {"corners": CornerTracker(image, first),
                "ring": RingTracker(image, first, ring_mask(image.shape, R0, t0, model, geo_points, camera))}
    estimates = {m: type("Pose", (), {"R": R0, "t": t0})() for m in METHODS}
    previous = {m: first for m in METHODS}
    alive = dict.fromkeys(METHODS, True)
    cover = first
    records = []
    for frame in range(anchor + 1, start + length):
        if frame >= len(frames):
            break
        k = frame - start + 1    # frames between the anchor and the loss are tracked, not scored
        row = table.loc[frame]
        cover = true_corners(row, model, camera, cover)
        image = paint_cover(frames[frame], cover) if painted and k >= 1 else frames[frame]
        for method in METHODS:
            speed = 0.0
            if method != "hold" and alive[method]:
                tracked = trackers[method].update(image)
                if method == "ring" and tracked is not None and jumped(previous[method], tracked, limit):
                    tracked = None
                estimate = None if tracked is None else pose_near(tracked, model, camera, estimates[method].R)
                if estimate is None:
                    alive[method] = False
                else:
                    speed = float(np.linalg.norm(tracked - previous[method], axis=2).mean())
                    estimates[method], previous[method] = estimate, tracked
            if not row["board_ok"] or k < 1:
                continue
            record = {"method": method, "k": k, "frame": frame, "lost": not alive[method], "e": None, "cov": None}
            if alive[method]:
                record["e"], record["cov"] = measure(estimates[method], speed, row, geo_points, model, camera,
                                                     speed_model)
            records.append(record)
    return records


def summarise(part):
    found = part[~part.lost]
    out = {"frames": int(len(part)), "lost": float(part.lost.mean())}
    if len(found):
        inside = (found.nees_growth >= CHI2_6_LOW) & (found.nees_growth <= CHI2_6_HIGH)
        out.update(median_rot_deg=float(found.rot_deg.median()), p95_rot_deg=float(found.rot_deg.quantile(0.95)),
                   median_trans_mm=float(found.trans_mm.median()), p95_trans_mm=float(found.trans_mm.quantile(0.95)),
                   nees_base_median=float(found.nees_base.median()), nees_median=float(found.nees_growth.median()),
                   anees=float(found.nees_growth.mean()), inside_95=float(inside.mean()))
    return out


def main():
    geo = carregar_geometria_medida()
    model = marker_model(geo["objp_marcador"])
    camera = load_camera(ROOT / "calib/camera.yaml")
    board_points = np.asarray(geo["objp_tabuleiro"], float)
    speed_model = json.loads((ROOT / "results/nees.json").read_text(encoding="utf-8"))["speed_model"]
    context = (board_points, model, camera, speed_model)

    records = []
    window_id = 0
    for name in ("E2", "E3"):
        table = load(name).set_index("frame", drop=False)
        frames = read_gray(RAW / f"{name}.webm")
        for length in WINDOWS:
            start = FIRST_FRAME
            while start + length < len(frames):
                if table.loc[start - 1, "marker_ok"]:
                    for r in run_window(frames, table, start, length, True, context):
                        r.update(video=name, window=window_id, length=length, painted=True)
                        records.append(r)
                    window_id += 1
                start += length + GAP_FRAMES
        print(name, "synthetic windows so far:", window_id, flush=True)

    table = load("E5").set_index("frame", drop=False)
    frames = read_gray(RAW / "E5.webm")
    missing = (~table.marker_ok).to_numpy()
    start = 1
    real_windows = 0
    while start < len(missing):
        if missing[start] and not missing[start - 1]:
            stop = start
            while stop + 1 < len(missing) and missing[stop + 1]:
                stop += 1
            if stop - start + 1 >= MIN_REAL_FRAMES:
                length = min(stop - start + 1, MAX_REAL_FRAMES)
                for r in run_window(frames, table, start, length, False, context):
                    r.update(video="E5", window=window_id, length=length, painted=False)
                    records.append(r)
                window_id += 1
                real_windows += 1
            start = stop + 1
        else:
            start += 1
    print("E5 real occlusions:", real_windows, flush=True)

    df = pd.DataFrame(records)
    df["fit"] = (df.video == "E2") & (df.window % 2 == 0)
    df["set"] = np.where(df.fit, "fit", np.where(df.video == "E2", "E2_test", df.video))
    found = df[~df.lost]
    errors = np.array(found.e.tolist())
    covs = np.array(found["cov"].tolist())
    df["rot_deg"], df["trans_mm"], df["nees_base"], df["nees_growth"] = np.nan, np.nan, np.nan, np.nan
    df.loc[found.index, "rot_deg"] = np.degrees(np.linalg.norm(errors[:, :3], axis=1))
    df.loc[found.index, "trans_mm"] = np.linalg.norm(errors[:, 3:], axis=1)

    growth = {}
    result = {"windows": WINDOWS, "methods": {}}
    for method in METHODS:
        here = (found.method == method).to_numpy()
        fit = here & found.fit.to_numpy()
        Q = fit_growth(found.k.to_numpy()[fit], errors[fit], np.diagonal(covs[fit], axis1=1, axis2=2), K_BINS)
        growth[method] = Q
        ids = found.index[here]
        df.loc[ids, "nees_base"] = [nees(e, c) for e, c in zip(errors[here], covs[here])]
        df.loc[ids, "nees_growth"] = [nees(e, c + k * np.diag(Q)) for e, c, k in
                                      zip(errors[here], covs[here], found.k.to_numpy()[here])]
        sd = np.sqrt(Q)
        entry = {"q_sd_per_sqrt_frame": {"rot_deg": np.degrees(sd[:3]).tolist(), "trans_mm": sd[3:].tolist()}, "sets": {}}
        part_all = df[df.method == method]
        for name, part in part_all.groupby("set"):
            bins = {}
            for interval, chunk in part.groupby(pd.cut(part.k, K_BINS), observed=True):
                bins[f"{interval.left + 1}-{interval.right}"] = summarise(chunk)
            entry["sets"][name] = {"all": summarise(part), "by_k": bins}
        result["methods"][method] = entry
        print(method, {k: {a: round(b, 2) for a, b in v["all"].items()} for k, v in entry["sets"].items()}, flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["e", "cov"]).to_parquet(OUT / "tracks.parquet", index=False)
    (ROOT / "results/occlusion.json").write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
