"""Marker corners and pose while the marker cannot be detected.

Three estimators, each started from the last detected frame:
  hold     repeat the last pose;
  corners  Lucas-Kanade on the 4 marker corners (fails when the corners are covered);
  ring     Lucas-Kanade on points of the target around the marker, plus a homography
           from their first positions that carries the 4 corners along.

The ring never uses the chessboard, which is the reference of the evaluation. Its mask
comes only from the last marker pose and the known layout of the target.
"""
import cv2
import numpy as np

from src.geometry import geodesic_rad
from src.pose import marker_poses

LK = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
FORWARD_BACKWARD_PX = 1.0
MIN_POINTS = 8
MIN_RING_POINTS = 20       # below this, or on a small patch, the homography degenerates
MIN_SPREAD = 1.0           # area of the hull of the ring points, in marker areas
RING_HALF_MM = 130.0
COVER_SCALE = 1.3    # the painted cover reaches this multiple of the marker half side
BOARD_MARGIN_MM = 25.0
MIN_SPEED_LIMIT = 4.0      # px/frame: tracked motion always allowed
SPEED_MARGIN = 3.0         # tracked motion allowed, as a multiple of the speed before the loss


def cover_polygon(corners):
    """Quad around the marker, as painted over it in the synthetic occlusion."""
    c = np.asarray(corners, float).reshape(4, 2)
    return (c.mean(0) + COVER_SCALE * (c - c.mean(0)) * 0.96).astype(np.int32)


def paint_cover(gray, corners, level=128):
    out = gray.copy()
    cv2.fillConvexPoly(out, cover_polygon(corners), int(level))
    return out


def _project_rect(x0, x1, y0, y1, R, t, camera):
    rect = np.array([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]], float)
    proj, _ = cv2.projectPoints(rect, cv2.Rodrigues(R)[0], t, camera.K, camera.dist)
    return proj.reshape(-1, 2).astype(np.int32)


def ring_mask(shape, R, t, model, board_points, camera):
    """Pixels around the marker on the target, without the marker and the board."""
    half = np.abs(model.points[:, :2]).max()
    mask = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(mask, _project_rect(-RING_HALF_MM, RING_HALF_MM, -RING_HALF_MM, RING_HALF_MM, R, t, camera), 255)
    # Board corners in the marker frame, grown by the outer squares of the board.
    board = (np.asarray(board_points, float) - model.t_target) @ model.R_target
    low, high = board[:, :2].min(0) - BOARD_MARGIN_MM, board[:, :2].max(0) + BOARD_MARGIN_MM
    cv2.fillConvexPoly(mask, _project_rect(low[0], high[0], low[1], high[1], R, t, camera), 0)
    cover = COVER_SCALE * half
    cv2.fillConvexPoly(mask, _project_rect(-cover, cover, -cover, cover, R, t, camera), 0)
    return mask


def _forward_backward(previous, current, points):
    """LK with a backward check; returns new points and the mask of the reliable ones."""
    forward, ok_f, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None, **LK)
    back, ok_b, _ = cv2.calcOpticalFlowPyrLK(current, previous, forward, None, **LK)
    error = np.linalg.norm(points - back, axis=2).ravel()
    return forward, ok_f.ravel().astype(bool) & ok_b.ravel().astype(bool) & (error < FORWARD_BACKWARD_PX)


class CornerTracker:
    """LK on the 4 marker corners, as in PLANO.md. Lost when a corner fails."""

    def __init__(self, gray, corners):
        self.gray = gray
        self.corners = np.asarray(corners, np.float32).reshape(-1, 1, 2)

    def update(self, gray):
        if self.corners is None:
            return None
        moved, good = _forward_backward(self.gray, gray, self.corners)
        self.gray = gray
        self.corners = moved if good.all() else None
        return self.corners


class RingTracker:
    """LK on points around the marker; the corners follow a homography from the first frame."""

    def __init__(self, gray, corners, mask):
        self.gray = gray
        self.start_corners = np.asarray(corners, np.float32).reshape(-1, 1, 2)
        found = cv2.goodFeaturesToTrack(gray, 300, 0.01, 7, mask=mask, blockSize=7)
        self.first = found
        self.points = found

    @property
    def n_points(self):
        return 0 if self.points is None else len(self.points)

    def update(self, gray):
        if self.n_points < MIN_POINTS:
            self.points = None
            return None
        moved, good = _forward_backward(self.gray, gray, self.points)
        self.gray = gray
        self.first, self.points = self.first[good], moved[good]
        if len(self.points) < MIN_POINTS:
            self.points = None
            return None
        H, inliers = cv2.findHomography(self.first, self.points, cv2.RANSAC, 2.0)
        if H is None or inliers.sum() < MIN_RING_POINTS:
            self.points = None
            return None
        keep = inliers.ravel().astype(bool)
        self.first, self.points = self.first[keep], self.points[keep]
        spread = cv2.contourArea(cv2.convexHull(self.first)) / cv2.contourArea(self.start_corners)
        if spread < MIN_SPREAD:
            self.points = None
            return None
        return cv2.perspectiveTransform(self.start_corners, H).astype(np.float32)


def speed_limit(recent_speed):
    """Fastest believable tracked motion, in px/frame, given the marker speed before the loss.

    A hand passing over the ring drags the points with it, and the tracker follows the
    hand. The target rarely starts moving much faster at the very moment it is covered.
    """
    return max(MIN_SPEED_LIMIT, SPEED_MARGIN * float(recent_speed))


def jumped(previous, tracked, limit, frames=1.0):
    """True if the corners moved faster than `limit` px/frame."""
    shift = np.linalg.norm(np.asarray(tracked, float) - np.asarray(previous, float), axis=2).mean()
    return shift / max(frames, 1e-6) > limit


def trusted_anchor(table, start, back=10, min_ratio=5.0, max_rms=0.5):
    """Latest frame before `start` whose detection is clean and not ambiguous.

    The frame right before an occlusion can have the hand over part of the marker, which
    gives a wrong pose with a low reprojection error. Uses only past detections.
    """
    for frame in range(start - 1, start - 1 - back, -1):
        row = table[frame] if frame in table else None
        if row is not None and row["marker_ok"] and row["ippe_ratio"] >= min_ratio and row["m_rms_px"] <= max_rms:
            return frame
    return start - 1


def pose_near(corners, model, camera, R_last):
    """IPPE solution closest to the last pose: during tracking, continuity beats reprojection."""
    solutions = marker_poses(corners, model, camera)
    if not solutions:
        return None
    return min(solutions, key=lambda p: geodesic_rad(p.R, R_last))


def fit_growth(k, errors, base_variance, edges):
    """Diagonal Q of the random walk var(e_k) = base + k Q.

    Tracking failures give rare huge errors, so the variance of each bin of k comes from
    the median of e^2 (0.4549 var for a Gaussian), and the line goes through the base.
    """
    k, errors = np.asarray(k, float), np.asarray(errors, float)
    base_variance = np.asarray(base_variance, float)
    centres, excess = [], []
    for low, high in zip(edges[:-1], edges[1:]):
        part = (k > low) & (k <= high)
        if part.sum() < 10:
            continue
        centres.append(np.median(k[part]))
        excess.append(np.median(errors[part] ** 2, axis=0) / 0.4549 - np.median(base_variance[part], axis=0))
    centres, excess = np.array(centres)[:, None], np.array(excess)
    return np.maximum((centres * excess).sum(0) / (centres ** 2).sum(), 0.0)
