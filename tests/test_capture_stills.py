"""Save rules of scripts/capture_stills.py, on rendered images of known pose."""
import importlib.util
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from render_target import TargetRenderer

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("capture_stills", ROOT / "scripts" / "capture_stills.py")
cs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cs)

K = cs.nominal_camera(1400)
LIMITS = cs.Limits(min_sharpness=20)
DT = 1 / cs.FPS


@pytest.fixture(scope="module")
def renderer():
    return TargetRenderer(K)


@pytest.fixture(scope="module")
def still(renderer):
    return renderer.render(*renderer.pose(tilt_deg=35, tilt_axis_deg=30))


def first_corners(image, renderer):
    return cs.evaluate(image, renderer.geo, K, LIMITS).corners


def test_still_sharp_board_passes_and_tilt_matches(renderer, still):
    previous = first_corners(still, renderer)
    check = cs.evaluate(still, renderer.geo, K, LIMITS, previous, DT)
    assert check.ok, check.problems
    assert check.tilt_deg == pytest.approx(35, abs=0.5)


def test_no_previous_frame_is_rejected(renderer, still):
    assert "sem quadro anterior" in cs.evaluate(still, renderer.geo, K, LIMITS).problems


def test_motion_is_rejected(renderer, still):
    moved = renderer.render(*renderer.pose(tilt_deg=35, tilt_axis_deg=30, center_px=(646, 480)))
    previous = first_corners(moved, renderer)
    check = cs.evaluate(still, renderer.geo, K, LIMITS, previous, DT)
    assert check.problems == ["em movimento"]
    assert check.speed == pytest.approx(6, abs=1)


def test_speed_uses_elapsed_time(renderer, still):
    moved = renderer.render(*renderer.pose(tilt_deg=35, tilt_axis_deg=30, center_px=(646, 480)))
    previous = first_corners(moved, renderer)
    check = cs.evaluate(still, renderer.geo, K, LIMITS, previous, 20 * DT)
    assert check.speed == pytest.approx(0.3, abs=0.05)
    assert check.ok


def test_blur_is_rejected(renderer, still):
    blurred = cv2.GaussianBlur(still, (0, 0), 2.5)
    previous = first_corners(blurred, renderer)
    assert previous is not None
    limits = cs.Limits(min_sharpness=cs.evaluate(still, renderer.geo, K, LIMITS).sharpness * 0.8)
    assert cs.evaluate(blurred, renderer.geo, K, limits, previous, DT).problems == ["imagem borrada"]


def test_partial_board_is_rejected(renderer):
    cut = renderer.render(*renderer.pose(center_px=(60, 480)))
    assert cs.evaluate(cut, renderer.geo, K, LIMITS).problems == ["tabuleiro incompleto"]


def test_saved_png_is_lossless_clean_and_never_overwrites(renderer, still, tmp_path):
    frame = cv2.cvtColor(still, cv2.COLOR_GRAY2BGR)
    previous = first_corners(still, renderer)
    check = cs.evaluate(still, renderer.geo, K, LIMITS, previous, DT)
    session = cs.Session(tmp_path, renderer.geo, K)
    original = frame.copy()
    cs.draw(frame, check, session, renderer.geo["padrao"], "")
    first = session.save(frame, check)
    assert np.array_equal(cv2.imread(str(first)), original)
    assert first.stat().st_size > frame.size   # compression level 0

    reopened = cs.Session(tmp_path, renderer.geo, K)
    assert len(reopened.photos) == 1
    assert reopened.save(frame, check).name == "still_0002.png"
    assert first.exists()


def test_counters_and_coverage(renderer, tmp_path):
    session = cs.Session(tmp_path, renderer.geo, K)
    assert session.coverage == 0
    for tilt, center in ((0, (320, 240)), (40, (960, 720))):
        image = renderer.render(*renderer.pose(tilt_deg=tilt, center_px=center, distance_mm=900))
        check = cs.evaluate(image, renderer.geo, K, LIMITS, first_corners(image, renderer), DT)
        session.save(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), check)
    assert len(session.photos) == 2
    assert session.n_tilted == 1
    expected = np.zeros((cs.GRID, cs.GRID), int)
    for photo in session.photos:
        cells = {(int(y * cs.GRID / cs.HEIGHT), int(x * cs.GRID / cs.WIDTH))
                 for x, y in photo["corners"].reshape(-1, 2)}
        for cell in cells:
            expected[cell] += 1
    assert np.array_equal(session.grid, expected)
    assert session.grid[0, 0] + session.grid[1, 1] >= 1
    assert session.grid[2, 2] + session.grid[3, 3] >= 1
    assert session.coverage == pytest.approx((expected > 0).sum() / cs.GRID ** 2)


class FakeCamera:
    """Replays one image; read() paces like a 30 fps camera."""

    def __init__(self, image):
        self.image = image

    def read(self):
        time.sleep(1 / 30)
        return True, self.image.copy()


def test_main_loop_saves_on_space_and_quits(renderer, still, tmp_path, monkeypatch):
    keys = iter([-1, ord(" "), -1, ord("q")])
    shown = []
    monkeypatch.setattr(cs.cv2, "namedWindow", lambda *a: None)
    monkeypatch.setattr(cs.cv2, "resizeWindow", lambda *a: None)
    monkeypatch.setattr(cs.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(cs.cv2, "getWindowProperty", lambda *a: 1)
    monkeypatch.setattr(cs.cv2, "imshow", lambda name, img: shown.append(img))
    monkeypatch.setattr(cs.cv2, "waitKey", lambda ms: next(keys))
    session = cs.Session(tmp_path, renderer.geo, K)
    cs.run(FakeCamera(cv2.cvtColor(still, cv2.COLOR_GRAY2BGR)), session, renderer.geo, K, LIMITS)
    assert len(session.photos) == 1
    assert (tmp_path / "still_0001.png").exists()
    assert len(shown) == 4
