"""Guidance and file output of scripts/record_video.py, without a camera."""
import csv
import importlib.util
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from render_target import TargetRenderer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("record_video", ROOT / "scripts" / "record_video.py")
rv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rv)
DT = 0.1


def text(guide):
    return " | ".join(t for t, _ in guide.lines() + guide.common_lines())


BOARD = np.stack(np.meshgrid(np.arange(8.0), np.arange(5.0)), -1).reshape(-1, 1, 2) * 40


def still(distance, tilt, shift=0.0):
    return rv.Sample(DT, True, True, tilt, distance, 0.1, "direita", corners=BOARD + shift)


def test_e1_counts_a_pose_only_after_ten_still_seconds():
    guide = rv.StillPoses()
    for _ in range(99):
        guide.update(still(0.5, 0))
    assert guide.index == 0 and "PARADO" in text(guide)
    guide.update(still(0.5, 0, shift=5.0))
    assert guide.held == 0 and "contagem reiniciada" in text(guide)
    for _ in range(102):
        guide.update(still(0.5, 0))
    assert guide.index == 1
    guide.update(still(0.5, 0))
    assert "Afaste a webcam 0.30 m" in text(guide)
    guide.update(still(1.2, 30))
    assert "Aproxime a webcam 0.40 m" in text(guide)
    guide.update(still(0.8, 5))
    assert "mais de lado" in text(guide)


def test_e1_ignores_detection_noise_on_a_resting_board():
    """Noise like the measured one (up to 0.6 px/frame) must not restart the count."""
    rng = np.random.default_rng(0)
    guide = rv.StillPoses()
    for _ in range(105):
        sample = still(0.5, 0)
        sample.corners = BOARD + rng.normal(0, 0.5, BOARD.shape)
        guide.update(sample)
    assert guide.index == 1


def test_e1_slow_drift_is_not_accepted_as_still():
    guide = rv.StillPoses()
    for i in range(102):
        guide.update(still(0.5, 0, shift=0.1 * i))
    assert guide.index == 0


def test_e1_short_detection_gap_keeps_the_count():
    guide = rv.StillPoses()
    for _ in range(50):
        guide.update(still(0.5, 0))
    held = guide.held
    for _ in range(3):
        guide.update(rv.Sample(DT, True, False))
    guide.update(still(0.5, 0))
    assert guide.held > held
    for _ in range(8):
        guide.update(rv.Sample(DT, True, False))
    assert guide.held == 0


def test_e1_finishes_after_five_poses():
    guide = rv.StillPoses()
    for distance, tilt in guide.poses:
        for _ in range(102):
            guide.update(still(distance, tilt))
    assert guide.done and "concluidas" in text(guide)


def test_e2_warns_when_too_fast_and_tracks_sides():
    guide = rv.SlowMotion()
    guide.update(rv.Sample(DT, True, True, 30, 0.6, 8.0, "topo"))
    assert "Rapido demais" in text(guide)
    guide.update(rv.Sample(DT, True, True, 30, 0.6, 1.0, "topo"))
    assert guide.sides["topo"] == pytest.approx(2 * DT)
    assert "Bom ritmo" in text(guide)


def test_e3_counts_strong_tilt_and_frontal_passes():
    guide = rv.StrongTilt()
    for _ in range(20):
        guide.update(rv.Sample(DT, True, True, 65, 0.6, 1.0, "topo"))
    assert guide.strong == pytest.approx(2.0) and "faixa certa" in text(guide)
    for tilt in (40, 5, 5, 40, 5):
        guide.update(rv.Sample(DT, True, True, tilt, 0.6, 1.0, "topo"))
    assert guide.frontal == 2
    guide.update(rv.Sample(DT, True, True, 80, 0.6, 1.0, "topo"))
    assert "passou de 75" in text(guide)


def test_e4_warns_when_the_reference_is_lost():
    guide = rv.Glare()
    for _ in range(60):
        guide.update(rv.Sample(DT, False, True))
    assert "Sem tabuleiro o E4 nao serve" in text(guide)
    assert "brilho" in text(guide).lower()


def occlude(guide, seconds, board=True):
    for _ in range(round(seconds / DT)):
        guide.update(rv.Sample(DT, board, False))
    guide.update(rv.Sample(DT, True, True))


def test_e5_only_counts_occlusions_that_follow_the_protocol():
    guide = rv.Occlusion()
    occlude(guide, 1.5)
    assert guide.count == 1
    occlude(guide, 0.3)
    occlude(guide, 4.0)
    occlude(guide, 1.5, board=False)
    assert guide.count == 1


def test_take_names_never_collide(tmp_path):
    assert rv.next_take(tmp_path, "E1").name == "E1_take01.mjpg"
    (tmp_path / "E1_take07.mjpg").write_bytes(b"x")
    assert rv.next_take(tmp_path, "E1").name == "E1_take08.mjpg"
    assert rv.next_take(tmp_path, "E2").name == "E2_take01.mjpg"


class JpegCamera:
    """Returns JPEG bytes at 30 fps, as the camera does with CONVERT_RGB off."""

    def __init__(self, image):
        self.jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].reshape(1, -1)

    def read(self):
        time.sleep(1 / 30)
        return True, self.jpeg


def test_recording_keeps_the_camera_jpeg_and_opens_in_opencv(tmp_path, monkeypatch):
    K = rv.nominal_camera(1420)
    renderer = TargetRenderer(K)
    image = cv2.cvtColor(renderer.render(*renderer.pose(20, 30)), cv2.COLOR_GRAY2BGR)
    camera = JpegCamera(image)
    keys = iter([-1, ord(" "), -1, -1, -1, ord(" "), ord("q")])
    for name, stub in (("namedWindow", lambda *a: None), ("destroyAllWindows", lambda: None),
                       ("getWindowProperty", lambda *a: 1), ("imshow", lambda *a: None),
                       ("waitKey", lambda ms: next(keys))):
        monkeypatch.setattr(rv.cv2, name, stub)
    rv.run(camera, rv.SlowMotion, tmp_path, renderer.geo, K)

    video = tmp_path / "E2_take01.mjpg"
    with open(video.with_suffix(".csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 3
    assert video.stat().st_size == camera.jpeg.size * len(rows)
    cap = cv2.VideoCapture(str(video))
    ok, frame = cap.read()
    assert ok and frame.shape == image.shape
    assert np.abs(frame.astype(int) - cv2.imdecode(camera.jpeg.ravel(), cv2.IMREAD_COLOR)).max() <= 2
