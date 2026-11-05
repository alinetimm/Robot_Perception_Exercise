"""calib/calibrate.py --fotos recovers a known camera from rendered photos."""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from alvo.referencia import carregar_geometria
from render_target import TargetRenderer, distortion_map

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "calib" / "calibrate.py"
sys.path.insert(0, str(ROOT / "calib"))
import calibrate as C  # noqa: E402
K_TRUE = np.array([[1200.0, 0, 650], [0, 1200, 470], [0, 0, 1]])
DIST_TRUE = np.array([-0.20, 0.10, 0, 0, 0])


def calibrate_cli(tmp_path, *source):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *source, "--yaml", str(tmp_path / "camera.yaml"),
         "--report", str(tmp_path / "relatorio.md"), "--frames-dir", str(tmp_path / "frames"),
         "--alvo-medido", str(tmp_path / "alvo_medido.json")],
        capture_output=True, text=True, cwd=ROOT)


@pytest.fixture(scope="module")
def photos(tmp_path_factory):
    folder = tmp_path_factory.mktemp("stills")
    renderer = TargetRenderer(K_TRUE)
    map_x, map_y = distortion_map(K_TRUE, DIST_TRUE)
    centers = [(x, y) for y in (280, 480, 680) for x in (340, 640, 940)]
    index = 0
    for tilt in (15, 40):
        for i, center in enumerate(centers):
            pose = renderer.pose(tilt, 40 * i + 20 * (tilt > 20), center, distance_mm=520 + 15 * i)
            image = cv2.remap(renderer.render(*pose), map_x, map_y, cv2.INTER_LINEAR, borderValue=110)
            index += 1
            cv2.imwrite(str(folder / f"still_{index:04d}.png"), image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    return folder


def test_recovers_known_intrinsics(photos, tmp_path):
    done = calibrate_cli(tmp_path, "--fotos", str(photos))
    assert done.returncode == 0, done.stderr
    cam = yaml.safe_load((tmp_path / "camera.yaml").read_text(encoding="utf-8"))
    K = np.array(cam["K"])
    k1, k2 = cam["dist_k1_k2_p1_p2_k3"][:2]
    assert (cam["image_width"], cam["image_height"]) == (1280, 960)
    assert cam["rms_px"] < 0.5
    assert K[0, 0] == pytest.approx(1200, rel=0.01)
    assert K[1, 1] == pytest.approx(1200, rel=0.01)
    assert K[0, 2] == pytest.approx(650, abs=5)
    assert K[1, 2] == pytest.approx(470, abs=5)
    assert k1 == pytest.approx(-0.20, abs=0.02)
    assert k2 == pytest.approx(0.10, abs=0.1)


def test_report_and_outputs(photos, tmp_path):
    (tmp_path / "frames").mkdir()
    stale = tmp_path / "frames" / "frame_00001.png"
    stale.write_bytes(b"old")
    assert calibrate_cli(tmp_path, "--fotos", str(photos)).returncode == 0
    report = (tmp_path / "relatorio.md").read_text(encoding="utf-8")
    assert "- Fotos: `" in report
    assert report.count("| still_") == 18   # different poses are all kept
    assert "Em movimento" not in report
    assert (tmp_path / "frames" / "mosaic.jpg").exists()
    assert not stale.exists()


def test_flat_target_measures_as_flat(photos, tmp_path):
    assert calibrate_cli(tmp_path, "--fotos", str(photos)).returncode == 0
    measured = json.loads((tmp_path / "alvo_medido.json").read_text(encoding="utf-8"))
    geo = carregar_geometria()
    assert np.abs(np.array(measured["cantos_tabuleiro_mm"]) - geo["objp_tabuleiro"]).max() < 0.3
    # The ArUco corner refinement pulls corners inward by a few tenths of a mm at this resolution.
    marker_error = np.abs(np.array(measured["cantos_marcador_mm"]) - geo["objp_marcador"])
    assert marker_error.mean() < 0.5 and marker_error.max() < 1.5


def test_bent_target_is_recovered_and_removes_the_bias_in_K():
    """Corners projected from a board bent by 2.5 mm, as measured on the real one."""
    geo = carregar_geometria()
    objp, objm = geo["objp_tabuleiro"], geo["objp_marcador"]

    def bend(points):
        bent = np.array(points, np.float64)
        bent[:, 2] = -2.5 * (1 - ((bent[:, 0] - 70) / 70) ** 2)
        return bent

    true_board, true_marker = bend(objp), bend(objm)
    rng = np.random.default_rng(1)
    frames = []
    for _ in range(80):
        rvec = rng.uniform(-0.9, 0.9, 3) * [1, 1, 0.3]
        tvec = np.array([rng.uniform(-150, 50), rng.uniform(-120, 0), rng.uniform(450, 900)])
        board, _ = cv2.projectPoints(true_board, rvec, tvec, K_TRUE, DIST_TRUE)
        marker, _ = cv2.projectPoints(true_marker, rvec, tvec, K_TRUE, DIST_TRUE)
        noise = rng.normal(0, 0.1, board.shape)
        frames.append({"corners": (board + noise).astype(np.float32), "marker": marker.astype(np.float32)})

    size, fixed = (1280, 960), geo["padrao"][0] - 1
    flat_K = C.calibrate(objp, frames, size)[1]
    rms, K, dist, *_, shape = C.calibrate_refined(objp, frames, size, fixed)
    assert rms < 0.2
    assert abs(K[0, 0] - 1200) < abs(flat_K[0, 0] - 1200)
    assert K[0, 0] == pytest.approx(1200, rel=0.005)
    assert np.abs(shape - true_board).max() < 0.2

    C.solve_board_poses(frames, shape, K, dist)
    assert np.abs(C.measure_marker(frames, K, dist) - true_marker).max() < 0.5


def test_duplicate_photo_is_dropped(photos, tmp_path):
    folder = tmp_path / "dup"
    folder.mkdir()
    for path in photos.glob("*.png"):
        (folder / path.name).write_bytes(path.read_bytes())
    (folder / "still_0100.png").write_bytes((photos / "still_0001.png").read_bytes())
    assert calibrate_cli(tmp_path, "--fotos", str(folder)).returncode == 0
    assert (tmp_path / "relatorio.md").read_text(encoding="utf-8").count("| still_") == 18


def test_photos_and_video_are_exclusive(tmp_path):
    done = calibrate_cli(tmp_path, "--fotos", "a", "--video", "b")
    assert done.returncode != 0
    assert "not allowed with" in done.stderr


def test_mixed_resolution_stops(photos, tmp_path):
    folder = tmp_path / "mixed"
    folder.mkdir()
    for path in sorted(photos.glob("*.png"))[:11]:
        (folder / path.name).write_bytes(path.read_bytes())
    cv2.imwrite(str(folder / "still_9999.png"), np.zeros((480, 640), np.uint8))
    done = calibrate_cli(tmp_path, "--fotos", str(folder))
    assert done.returncode != 0
    assert "one camera mode" in done.stderr
