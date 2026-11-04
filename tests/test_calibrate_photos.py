"""calib/calibrate.py --fotos recovers a known camera from rendered photos."""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from render_target import TargetRenderer, distortion_map

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "calib" / "calibrate.py"
K_TRUE = np.array([[1200.0, 0, 650], [0, 1200, 470], [0, 0, 1]])
DIST_TRUE = np.array([-0.20, 0.10, 0, 0, 0])


def calibrate_cli(tmp_path, *source):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *source, "--yaml", str(tmp_path / "camera.yaml"),
         "--report", str(tmp_path / "relatorio.md"), "--frames-dir", str(tmp_path / "frames")],
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
