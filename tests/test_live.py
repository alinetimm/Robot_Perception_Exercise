"""State machine, ellipse and on-screen check of the live demo, on a rendered sequence."""
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from alvo.referencia import carregar_geometria
from render_target import TargetRenderer
from src.geometry import geodesic_rad
from src.live import DETECTED, LOST, TRACKED, Estimate, LiveEstimator, reference_check, tip_ellipse
from src.occlusion import paint_cover
from src.pose import Camera, expected_marker_pose, marker_model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("live_demo", ROOT / "scripts" / "live_demo.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)

K = np.array([[1420.0, 0, 632], [0, 1420, 469], [0, 0, 1]])
CAM = Camera(K, np.zeros(5))
GEO = carregar_geometria()
MODEL = marker_model(GEO["objp_marcador"])
SPEED = {"base_px": 0.33, "gain": 0.07, "board_ratio": 0.87}
GROWTH = np.r_[np.radians([0.24, 0.16, 0.06]), [0.11, 0.27, 0.47]] ** 2
DT = 1 / 30


@pytest.fixture(scope="module")
def frames():
    r = TargetRenderer(K)
    texture = cv2.GaussianBlur(np.random.default_rng(0).integers(0, 255, r.page.shape).astype(np.float32), (0, 0), 2.5)
    # Texture on the paper, but with the white margin the real marker has around it.
    corners = cv2.perspectiveTransform(MODEL.points[None, :, :2] * 1.35 + MODEL.t_target[:2], r.board_to_page)
    margin = cv2.fillConvexPoly(np.zeros_like(r.page), corners.reshape(-1, 2).astype(np.int32), 1) > 0
    blank = (r.page > 240) & ~margin
    r.page = np.where(blank, np.clip(190 + (texture - texture.mean()) * 10, 90, 250), r.page).astype(np.uint8)
    out = []
    for i in range(45):
        R, t = r.pose(20 + 0.1 * i, 30, (640 + i, 300), 620)
        R_m, t_m = expected_marker_pose(type("P", (), {"R": R, "t": t}), MODEL)
        corners = cv2.projectPoints(MODEL.points, cv2.Rodrigues(R_m)[0], t_m, K, None)[0].astype(np.float32)
        out.append((r.render(R, t), corners, R_m, t_m))
    return out


def estimator():
    return LiveEstimator(GEO, MODEL, CAM, SPEED, GROWTH)


def test_states_through_an_occlusion(frames):
    live = estimator()
    seen = []
    for i, (image, corners, R_true, _) in enumerate(frames):
        covered = 15 <= i < 35
        estimate = live.step(paint_cover(image, corners) if covered else image, i * DT)
        seen.append(estimate)
        assert estimate.state == (TRACKED if covered else DETECTED)
        assert geodesic_rad(estimate.R, R_true) < np.radians(3)
    rot = [np.trace(e.cov[:3, :3]) for e in seen[15:35]]
    assert rot[-1] > 3 * rot[0]
    assert seen[34].frames_lost == pytest.approx(20, abs=1)
    assert np.trace(seen[35].cov[:3, :3]) < rot[-1]


def test_lost_when_nothing_can_be_tracked(frames):
    live = estimator()
    for i, (image, *_) in enumerate(frames[:5]):
        live.step(image, i * DT)
    blank = np.full_like(frames[0][0], 128)
    assert live.step(blank, 5 * DT).state == LOST
    assert live.step(blank, 6 * DT).state == LOST
    assert live.step(frames[7][0], 7 * DT).state == DETECTED


def test_lost_before_any_detection(frames):
    assert estimator().step(np.full_like(frames[0][0], 128), 0.0).state == LOST


def test_tip_ellipse_of_a_known_covariance():
    R, t = np.eye(3), np.array([0.0, 0.0, 1000.0])
    cov = np.diag([0, 0, 0, 1.0, 4.0, 0.0])    # 1 mm in x, 2 mm in y
    centre, axes, angle = tip_ellipse(R, t, cov, CAM, length_mm=100, n_sigma=3, gain=2)
    assert centre == pytest.approx([632, 469], abs=0.01)
    scale = 1420 / 900
    assert axes == pytest.approx([2 * 3 * 2 * scale, 2 * 3 * 1 * scale], rel=1e-3)
    assert abs(angle) % 180 == pytest.approx(90, abs=1e-3)


def test_tip_ellipse_grows_with_rotation_uncertainty():
    R, t = np.eye(3), np.array([0.0, 0.0, 700.0])
    small = tip_ellipse(R, t, np.eye(6) * 1e-6, CAM)[1]
    large = tip_ellipse(R, t, np.diag([1e-4, 1e-4, 1e-6, 1e-6, 1e-6, 1e-6]), CAM)[1]
    assert large[0] > 3 * small[0]


def test_reference_check_on_a_good_estimate(frames):
    image, corners, R_m, t_m = frames[0]
    estimate = estimator().step(image, 0.0)
    check = reference_check(image, estimate, GEO, MODEL, CAM, SPEED)
    assert check["rot_deg"] < 2 and check["trans_mm"] < 5 and check["nees"] < 50


def test_reference_check_without_a_board(frames):
    estimate = estimator().step(frames[0][0], 0.0)
    blank = np.full_like(frames[0][0], 128)
    assert reference_check(blank, estimate, GEO, MODEL, CAM, SPEED) is None


def test_draw_every_state(frames):
    image, *_ = frames[0]
    live = estimator()
    estimate = live.step(image, 0.0)
    colour = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    check = {"rot_deg": 0.5, "trans_mm": 1.2, "nees": 4.0}
    for state in (estimate, Estimate(TRACKED, estimate.corners, estimate.R, estimate.t, estimate.cov, 0.3, 12),
                  Estimate(LOST)):
        view = demo.draw(colour.copy(), state, CAM, check if state.R is not None else None, 5.0, 28.0, True)
        assert view.shape == colour.shape
        assert np.abs(view.astype(int) - colour).sum() > 0


def test_huge_covariance_does_not_break_the_ellipse(frames):
    image, *_ = frames[0]
    estimate = estimator().step(image, 0.0)
    estimate.cov = estimate.cov + np.eye(6) * 1e6
    demo.draw(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), estimate, CAM, None, 5.0, 28.0, False)


def test_screen_recording_keeps_real_time(tmp_path, frames):
    colour = cv2.cvtColor(frames[0][0], cv2.COLOR_GRAY2BGR)
    recorder = demo.ScreenRecorder(tmp_path / "demo_take01.mp4")
    recorder.start -= 1.0    # as if the first frame arrived one second after the start
    recorder.add(colour)
    recorder.close()
    cap = cv2.VideoCapture(str(tmp_path / "demo_take01.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == pytest.approx(demo.FPS, abs=2)
    assert demo.next_recording(tmp_path).name == "demo_take02.mp4"


def test_contaminated_last_detection_does_not_end_the_track(frames):
    """A hand can bend the last detected corners; the motion check must start at the anchor."""
    live = estimator()
    image, corners, *_ = frames[0]
    for i in range(6):
        live.step(image, i * DT)
    live.history[-1]["ippe_ratio"] = 1.1        # distrusted, as with a hand over the marker
    live.history[-1]["corners"] = corners + np.float32([12, 9])
    live.previous_corners = live.history[-1]["corners"]
    assert live.step(paint_cover(image, corners), 6 * DT).state == TRACKED


def test_tracker_dragged_away_from_a_still_target_is_lost(frames):
    """Seen in the recorded demo: the ring followed a hand while the board stood still."""
    live = estimator()
    image, corners, *_ = frames[0]
    for i in range(5):
        live.step(image, i * DT)

    class Dragged:
        def update(self, gray):
            return corners + np.float32([0, -40])

    live.tracker, live.lost_stamp = Dragged(), 5 * DT
    assert live.step(paint_cover(image, corners), 5 * DT).state == LOST
