"""Rotations and pose errors.

Pose convention: X_cam = R X_target + t. Internal angles are radians.
Perturbation (CLAUDE.md): R <- exp([dtheta]x) R, t <- t + dt, with the error vector
ordered [dtheta, dt].
"""
import cv2
import numpy as np


def exp_so3(w):
    return cv2.Rodrigues(np.asarray(w, float).reshape(3, 1))[0]


def log_so3(R):
    return cv2.Rodrigues(np.asarray(R, float))[0].ravel()


def geodesic_rad(R_a, R_b):
    """||log(R_a^T R_b)||."""
    return float(np.linalg.norm(log_so3(R_a.T @ R_b)))


def pose_error(R_est, t_est, R_ref, t_ref):
    """Error e = [dtheta, dt] such that R_est = exp([dtheta]x) R_ref, t_est = t_ref + dt."""
    return np.concatenate([log_so3(R_est @ R_ref.T), np.ravel(t_est) - np.ravel(t_ref)])
