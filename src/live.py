"""Live marker pose with its covariance: detection, then tracking, then loss.

Everything here uses the marker and past frames only, as a real landing would. The
board enters only in `reference_check`, which measures the error for display.
"""
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from alvo.referencia import cantos_marcador, cantos_tabuleiro
from src.covariance import jacobian, nees, pose_covariance, sigma_at_speed, transfer_to_marker
from src.geometry import pose_error
from src.occlusion import RingTracker, jumped, pose_near, ring_mask, speed_limit, trusted_anchor
from src.pose import board_pose, expected_marker_pose, marker_poses

DETECTED, TRACKED, LOST = "DETECTADO", "RASTREADO", "PERDIDO"
HISTORY_FRAMES = 12
NOMINAL_FPS = 30.0
AMBIGUOUS_RATIO = 5.0


@dataclass
class Estimate:
    state: str
    corners: np.ndarray | None = None
    R: np.ndarray | None = None
    t: np.ndarray | None = None
    cov: np.ndarray | None = None
    sigma_px: float = 0.0
    frames_lost: float = 0.0


class LiveEstimator:
    def __init__(self, geo, model, camera, speed_model, growth):
        self.geo, self.model, self.camera = geo, model, camera
        self.speed_model, self.growth = speed_model, np.diag(growth)
        self.board_points = np.asarray(geo["objp_tabuleiro"], float)
        self.frame = -1
        self.history = deque(maxlen=HISTORY_FRAMES)
        self.previous_corners = self.previous_stamp = self.lost_stamp = None
        self.R_last = None
        self.tracker = None
        self.recent_speeds = deque(maxlen=HISTORY_FRAMES)

    def _covariance(self, R, t, corners, stamp):
        speed = 0.0
        if self.previous_corners is not None and stamp > self.previous_stamp:
            shift = np.linalg.norm(corners - self.previous_corners, axis=2).mean()
            speed = float(shift) / ((stamp - self.previous_stamp) * NOMINAL_FPS)
        sigma = float(sigma_at_speed(speed, self.speed_model["base_px"], self.speed_model["gain"]))
        self.previous_corners, self.previous_stamp = corners, stamp
        self.last_speed = speed
        return sigma ** 2 * pose_covariance(jacobian(self.model.points, R, t, self.camera), 1.0), sigma

    def _detect(self, gray):
        corners = cantos_marcador(gray, self.geo)
        solutions = [] if corners is None else marker_poses(corners, self.model, self.camera)
        if not solutions:
            return None
        ratio = solutions[1].rms_px / max(solutions[0].rms_px, 1e-9) if len(solutions) > 1 else np.inf
        best = solutions[0]
        if self.R_last is not None and ratio < AMBIGUOUS_RATIO:
            best = pose_near(corners, self.model, self.camera, self.R_last)
        return corners, best, ratio

    def _start_tracking(self, stamp):
        """Rebuild the track from the last trusted frame, replaying the frames kept since."""
        table = {h["frame"]: h for h in self.history}
        anchor = table[trusted_anchor(table, self.frame)] if self.frame - 1 in table else self.history[-1]
        mask = ring_mask(anchor["gray"].shape, anchor["R"], anchor["t"], self.model, self.board_points, self.camera)
        self.tracker = RingTracker(anchor["gray"], anchor["corners"], mask)
        self.R_last = anchor["R"]
        self.lost_stamp = stamp
        # The frames after the anchor may hold the hand: the motion check starts at the anchor.
        self.previous_corners, self.previous_stamp = anchor["corners"], anchor["stamp"]
        for h in self.history:
            if h["frame"] > anchor["frame"]:
                self.tracker.update(h["gray"])

    def step(self, gray, stamp):
        self.frame += 1
        found = self._detect(gray)
        if found is not None:
            corners, pose, ratio = found
            self.tracker, self.R_last = None, pose.R
            cov, sigma = self._covariance(pose.R, pose.t, corners, stamp)
            self.recent_speeds.append(self.last_speed)
            self.history.append({"frame": self.frame, "stamp": stamp, "gray": gray, "corners": corners,
                                 "R": pose.R, "t": pose.t,
                                 "marker_ok": True, "ippe_ratio": ratio, "m_rms_px": pose.rms_px})
            return Estimate(DETECTED, corners, pose.R, pose.t, cov, sigma)

        if self.tracker is None:
            if not self.history or self.lost_stamp is not None and self.history[-1]["frame"] < self.frame - 1:
                return Estimate(LOST)
            self._start_tracking(stamp)
        corners = self.tracker.update(gray)
        if corners is not None and self.previous_corners is not None:
            frames = max((stamp - self.previous_stamp) * NOMINAL_FPS, 1.0)
            if jumped(self.previous_corners, corners, speed_limit(np.median(self.recent_speeds)), frames):
                corners = None
        pose = None if corners is None else pose_near(corners, self.model, self.camera, self.R_last)
        if pose is None:
            self.tracker = None
            return Estimate(LOST)
        self.R_last = pose.R
        cov, sigma = self._covariance(pose.R, pose.t, corners, stamp)
        k = (stamp - self.lost_stamp) * NOMINAL_FPS + 1
        return Estimate(TRACKED, corners, pose.R, pose.t, cov + k * self.growth, sigma, k)


def tip_ellipse(R, t, cov, camera, length_mm=100.0, n_sigma=3.0, gain=1.0):
    """Image ellipse (centre, semi-axes, angle in degrees) of the tip of the marker normal.

    The centre of the marker hardly moves under rotation, so its ellipse would be too
    small to see. The tip of the normal shows the uncertainty of the marker orientation.
    """
    tip = np.array([[0.0, 0.0, -length_mm]])
    centre = cv2.projectPoints(tip, cv2.Rodrigues(R)[0], t, camera.K, camera.dist)[0].ravel()
    J = jacobian(tip, R, t, camera)
    values, vectors = np.linalg.eigh(J @ cov @ J.T)
    axes = gain * n_sigma * np.sqrt(np.maximum(values[::-1], 0.0))
    return centre, axes, float(np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1])))


def reference_check(gray, estimate, geo, model, camera, speed_model):
    """Error and NEES of one estimate against the board of the same frame; None if no board."""
    corners = cantos_tabuleiro(gray, geo)
    board_points = np.asarray(geo["objp_tabuleiro"], float)
    reference = None if corners is None else board_pose(corners, board_points, camera)
    if reference is None:
        return None
    R_exp, t_exp = expected_marker_pose(reference, model)
    e = pose_error(estimate.R, estimate.t, R_exp, t_exp)
    sigma_board = estimate.sigma_px * speed_model["board_ratio"]
    cov = estimate.cov + transfer_to_marker(
        pose_covariance(jacobian(board_points, reference.R, reference.t, camera), sigma_board),
        reference.R, model.t_target)
    return {"rot_deg": float(np.degrees(np.linalg.norm(e[:3]))), "trans_mm": float(np.linalg.norm(e[3:])),
            "nees": nees(e, cov)}
