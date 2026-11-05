"""Per-frame poses of the reference board and of the marker."""
from dataclasses import dataclass

import cv2
import numpy as np
import yaml

from alvo.referencia import cantos_marcador, cantos_tabuleiro
from src.geometry import pose_error


@dataclass
class Camera:
    K: np.ndarray
    dist: np.ndarray


@dataclass
class Pose:
    R: np.ndarray
    t: np.ndarray
    rms_px: float


@dataclass
class MarkerModel:
    """Marker frame: origin at the marker centre, axes along the target's, z=0 on its plane."""
    points: np.ndarray        # (4, 3) corners in the marker frame
    R_target: np.ndarray      # X_target = R_target X_marker + t_target
    t_target: np.ndarray


def load_camera(path):
    data = yaml.safe_load(open(path, encoding="utf-8"))
    return Camera(np.array(data["K"], float), np.array(data["dist_k1_k2_p1_p2_k3"], float))


def marker_model(corners_target):
    """Fit the marker frame to its 4 corners (TL, TR, BR, BL) given in the target frame.

    The measured corners are used instead of an ideal 80 mm square: they absorb the
    bend of the sheet and the inward bias of the corner detector.
    """
    c = np.asarray(corners_target, float)
    centre = c.mean(0)
    normal = np.linalg.svd(c - centre)[2][2]
    normal = normal * np.sign(normal[2])
    x = (c[1] - c[0]) + (c[2] - c[3])
    x = x - normal * (x @ normal)
    x /= np.linalg.norm(x)
    R = np.column_stack([x, np.cross(normal, x), normal])
    points = (c - centre) @ R
    points[:, 2] = 0.0
    return MarkerModel(points, R, centre)


def reprojection_rms(points, image_points, R, t, camera):
    proj, _ = cv2.projectPoints(np.asarray(points, float), cv2.Rodrigues(R)[0], t, camera.K, camera.dist)
    diff = proj.reshape(-1, 2) - np.asarray(image_points, float).reshape(-1, 2)
    return float(np.sqrt((diff ** 2).sum(1).mean()))


def solve_in_front(points, image_points, K, dist):
    """solvePnP that never returns a target behind the camera.

    For a planar target, (r1, r2, t) and (-r1, -r2, -t) project to the same image, and
    solvePnP does return the second one, mostly for far, frontal targets.
    """
    points, image_points = np.asarray(points, float), np.asarray(image_points, float)
    ok, rvec, tvec = cv2.solvePnP(points, image_points, K, dist)
    if ok and tvec[2, 0] < 0:
        mirrored = cv2.Rodrigues(cv2.Rodrigues(rvec)[0] @ np.diag([-1.0, -1.0, 1.0]))[0]
        ok, rvec, tvec = cv2.solvePnP(points, image_points, K, dist, mirrored, -tvec, useExtrinsicGuess=True)
        ok = ok and tvec[2, 0] > 0
    return ok, rvec, tvec


def board_pose(corners, board_points, camera):
    ok, rvec, tvec = solve_in_front(board_points, corners, camera.K, camera.dist)
    if not ok:
        return None
    R, t = cv2.Rodrigues(rvec)[0], tvec.ravel()
    return Pose(R, t, reprojection_rms(board_points, corners, R, t, camera))


def marker_poses(corners, model, camera):
    """Both IPPE solutions, lowest reprojection error first."""
    n, rvecs, tvecs, _ = cv2.solvePnPGeneric(model.points, corners.astype(float), camera.K, camera.dist,
                                             flags=cv2.SOLVEPNP_IPPE)
    poses = []
    for rvec, tvec in zip(rvecs[:n], tvecs[:n]):
        R, t = cv2.Rodrigues(rvec)[0], tvec.ravel()
        poses.append(Pose(R, t, reprojection_rms(model.points, corners, R, t, camera)))
    return sorted(poses, key=lambda p: p.rms_px)


def expected_marker_pose(reference, model):
    """Marker pose implied by the board pose and the known marker placement."""
    return reference.R @ model.R_target, reference.R @ model.t_target + reference.t


def tilt_deg(R):
    """Angle between the target normal and the optical axis."""
    return float(np.degrees(np.arccos(np.clip(abs(R[2, 2]), 0, 1))))


def process_frame(gray, geo, model, camera):
    """Everything measured in one frame, as a flat dict of scalars and corner arrays."""
    row = {"board_ok": False, "marker_ok": False}
    board = cantos_tabuleiro(gray, geo)
    marker = cantos_marcador(gray, geo)
    row["board_corners"], row["marker_corners"] = board, marker
    reference = None
    if board is not None:
        reference = board_pose(board, geo["objp_tabuleiro"], camera)
    if reference is not None:
        row.update(board_ok=True, ref_R=reference.R, ref_t=reference.t, ref_rms_px=reference.rms_px,
                   tilt_deg=tilt_deg(reference.R), distance_mm=float(np.linalg.norm(reference.t)))
    if marker is None:
        return row
    solutions = marker_poses(marker, model, camera)
    if not solutions:
        return row
    row.update(marker_ok=True, m_R=solutions[0].R, m_t=solutions[0].t, m_rms_px=solutions[0].rms_px)
    if len(solutions) > 1:
        second = solutions[1]
        row.update(m2_R=second.R, m2_t=second.t, m2_rms_px=second.rms_px,
                   ippe_ratio=second.rms_px / max(solutions[0].rms_px, 1e-9))
    if reference is not None:
        R_exp, t_exp = expected_marker_pose(reference, model)
        row["err"] = pose_error(solutions[0].R, solutions[0].t, R_exp, t_exp)
        if len(solutions) > 1:
            row["err2"] = pose_error(solutions[1].R, solutions[1].t, R_exp, t_exp)
    return row
