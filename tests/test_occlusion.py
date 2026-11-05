"""Tracking through an occlusion on a rendered sequence of known motion."""
import cv2
import numpy as np
import pytest

from alvo.referencia import carregar_geometria
from render_target import TargetRenderer
from src.geometry import geodesic_rad
from src.occlusion import (CornerTracker, RingTracker, cover_polygon, fit_growth, paint_cover, pose_near,
                           ring_mask, speed_limit, jumped, trusted_anchor)
from src.pose import Camera, expected_marker_pose, marker_model

K = np.array([[1420.0, 0, 632], [0, 1420, 469], [0, 0, 1]])
CAM = Camera(K, np.zeros(5))
GEO = carregar_geometria()
MODEL = marker_model(GEO["objp_marcador"])


@pytest.fixture(scope="module")
def renderer():
    """The target with texture on its blank paper, like the wood around the real sheet."""
    r = TargetRenderer(K)
    texture = cv2.GaussianBlur(np.random.default_rng(0).integers(0, 255, r.page.shape).astype(np.float32), (0, 0), 2.5)
    texture = (texture - texture.mean()) * 10
    blank = r.page > 240
    r.page = np.where(blank, np.clip(190 + texture, 90, 250), r.page).astype(np.uint8)
    return r


def sequence(renderer, frames=31, speed_px=2.0, tilt=20.0):
    """Images and true marker corners of a target sliding and turning slowly."""
    out = []
    for i in range(frames):
        R, t = renderer.pose(tilt + 0.1 * i, 30, (640 + speed_px * i, 300), 620)
        R_m, t_m = expected_marker_pose(type("P", (), {"R": R, "t": t}), MODEL)
        corners = cv2.projectPoints(MODEL.points, cv2.Rodrigues(R_m)[0], t_m, K, None)[0].astype(np.float32)
        out.append((renderer.render(R, t), corners, R_m, t_m))
    return out


def test_cover_hides_the_whole_marker(renderer):
    image, corners, *_ = sequence(renderer, frames=1)[0]
    covered = paint_cover(image, corners)
    x0, y0 = corners.reshape(-1, 2).min(0).astype(int)
    x1, y1 = corners.reshape(-1, 2).max(0).astype(int)
    assert (covered[y0:y1, x0:x1] == 128).all()
    assert cv2.pointPolygonTest(cover_polygon(corners).astype(np.float32), tuple(map(float, corners[0, 0])), True) > 5


def test_ring_mask_excludes_marker_and_board(renderer):
    image, corners, R_m, t_m = sequence(renderer, frames=1)[0]
    mask = ring_mask(image.shape, R_m, t_m, MODEL, GEO["objp_tabuleiro"], CAM)
    assert mask.sum() > 0
    assert mask[cv2.fillConvexPoly(np.zeros_like(mask), cover_polygon(corners), 1) > 0].sum() == 0
    R_ref, t_ref = R_m @ MODEL.R_target.T, t_m - R_m @ MODEL.R_target.T @ MODEL.t_target
    board = cv2.projectPoints(np.asarray(GEO["objp_tabuleiro"], float), cv2.Rodrigues(R_ref)[0], t_ref, K, None)[0]
    x, y = np.rint(board.reshape(-1, 2)).astype(int).T
    assert mask[y, x].sum() == 0


def test_ring_tracker_follows_a_covered_marker(renderer):
    frames = sequence(renderer)
    image, corners, R_m, t_m = frames[0]
    tracker = RingTracker(image, corners, ring_mask(image.shape, R_m, t_m, MODEL, GEO["objp_tabuleiro"], CAM))
    assert tracker.n_points >= 20
    for image, truth, R_true, t_true in frames[1:]:
        estimate = tracker.update(paint_cover(image, truth))
        assert estimate is not None
    assert np.linalg.norm(estimate - truth, axis=2).max() < 1.5
    pose = pose_near(estimate, MODEL, CAM, frames[0][2])
    assert geodesic_rad(pose.R, R_true) < np.radians(2)


def test_corner_tracker_cannot_follow_a_covered_marker(renderer):
    """Why the ring exists: this is the method of the plan under a full occlusion."""
    frames = sequence(renderer)
    tracker = CornerTracker(frames[0][0], frames[0][1])
    estimate = None
    for image, truth, *_ in frames[1:]:
        estimate = tracker.update(paint_cover(image, truth))
    final_error = np.inf if estimate is None else np.linalg.norm(estimate - truth, axis=2).max()
    assert final_error > 5


def test_corner_tracker_follows_a_visible_marker(renderer):
    frames = sequence(renderer, frames=11)
    tracker = CornerTracker(frames[0][0], frames[0][1])
    for image, truth, *_ in frames[1:]:
        estimate = tracker.update(image)
    assert np.linalg.norm(estimate - truth, axis=2).max() < 1.5


def test_trackers_report_loss(renderer):
    image, corners, R_m, t_m = sequence(renderer, frames=1)[0]
    tracker = RingTracker(image, corners, ring_mask(image.shape, R_m, t_m, MODEL, GEO["objp_tabuleiro"], CAM))
    assert tracker.update(np.full_like(image, 128)) is None
    assert tracker.update(image) is None


def test_fit_growth_recovers_a_random_walk():
    rng = np.random.default_rng(0)
    q = np.array([1e-6, 4e-6, 1e-6, 0.5, 0.2, 2.0])
    base = q / 10
    k = rng.integers(1, 60, 40000)
    errors = rng.normal(0, 1, (len(k), 6)) * np.sqrt(base + k[:, None] * q)
    edges = [0, 5, 15, 30, 60]
    assert fit_growth(k, errors, np.tile(base, (len(k), 1)), edges) == pytest.approx(q, rel=0.1)


def test_fit_growth_ignores_rare_tracking_failures():
    rng = np.random.default_rng(1)
    q = np.array([1e-6, 1e-6, 1e-6, 0.5, 0.5, 0.5])
    k = rng.integers(1, 60, 20000)
    errors = rng.normal(0, 1, (len(k), 6)) * np.sqrt(k[:, None] * q)
    errors[::50] *= 100
    assert fit_growth(k, errors, np.zeros((len(k), 6)), [0, 5, 15, 30, 60]) == pytest.approx(q, rel=0.2)


def test_ring_reports_loss_when_few_points_remain(renderer):
    frames = sequence(renderer, frames=3)
    image, corners, R_m, t_m = frames[0]
    tracker = RingTracker(image, corners, ring_mask(image.shape, R_m, t_m, MODEL, GEO["objp_tabuleiro"], CAM))
    tracker.first, tracker.points = tracker.first[:15], tracker.points[:15]
    assert tracker.update(frames[1][0]) is None


def test_ring_reports_loss_when_the_points_are_bunched(renderer):
    frames = sequence(renderer, frames=3)
    image, corners, R_m, t_m = frames[0]
    tracker = RingTracker(image, corners, ring_mask(image.shape, R_m, t_m, MODEL, GEO["objp_tabuleiro"], CAM))
    centre = tracker.first.reshape(-1, 2).mean(0)
    nearest = np.argsort(np.linalg.norm(tracker.first.reshape(-1, 2) - centre, axis=1))[:25]
    tracker.first, tracker.points = tracker.first[nearest], tracker.points[nearest]
    tracker.start_corners = tracker.start_corners * 4   # a marker much larger than the patch
    assert tracker.update(frames[1][0]) is None


def test_trusted_anchor_skips_a_contaminated_last_frame():
    rows = {i: {"marker_ok": True, "ippe_ratio": 20.0, "m_rms_px": 0.1} for i in range(90, 100)}
    rows[99] = {"marker_ok": True, "ippe_ratio": 1.2, "m_rms_px": 0.1}
    rows[98] = {"marker_ok": True, "ippe_ratio": 20.0, "m_rms_px": 2.0}
    rows[97] = {"marker_ok": False, "ippe_ratio": np.nan, "m_rms_px": np.nan}
    assert trusted_anchor(rows, 100) == 96
    assert trusted_anchor({99: {"marker_ok": True, "ippe_ratio": 1.0, "m_rms_px": 3.0}}, 100) == 99


def test_motion_gate():
    assert speed_limit(0.1) == 4.0
    assert speed_limit(5.0) == 15.0
    a = np.zeros((4, 1, 2))
    assert jumped(a, a + [[10.0, 0.0]], 4.0)
    assert not jumped(a, a + [[3.0, 0.0]], 4.0)
    assert not jumped(a, a + [[10.0, 0.0]], 4.0, frames=3)
