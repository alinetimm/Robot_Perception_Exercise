"""First-order pose covariance and NEES.

The Jacobian is taken by central differences with respect to the local perturbation
R <- exp([dtheta]x) R, t <- t + dt, in the order [dtheta, dt], the same side and order
as src.geometry.pose_error.
"""
import cv2
import numpy as np
from scipy.optimize import minimize

from src.geometry import exp_so3

EPS_ROT = 1e-4    # rad; cv2.Rodrigues rounds rotations below 1e-5 rad to zero
EPS_TRANS = 1e-4  # mm

# Two-sided 95% interval of a single chi-square sample with 6 degrees of freedom.
CHI2_6_LOW, CHI2_6_HIGH = 1.2373, 14.4494
CHI2_6_MEDIAN = 5.3481
DOF = 6


def project(points, R, t, camera):
    proj, _ = cv2.projectPoints(points, cv2.Rodrigues(R)[0], t, camera.K, camera.dist)
    return proj.ravel()


def jacobian(points, R, t, camera):
    """d(projection)/d[dtheta, dt], shape (2N, 6)."""
    points = np.asarray(points, float)
    J = np.empty((2 * len(points), 6))
    for i in range(3):
        step = np.zeros(3)
        step[i] = EPS_ROT
        J[:, i] = (project(points, exp_so3(step) @ R, t, camera)
                   - project(points, exp_so3(-step) @ R, t, camera)) / (2 * EPS_ROT)
        step[i] = EPS_TRANS
        J[:, 3 + i] = (project(points, R, t + step, camera)
                       - project(points, R, t - step, camera)) / (2 * EPS_TRANS)
    return J


def pose_covariance(J, sigma_px):
    """Sigma = sigma^2 (J^T J)^-1, on a scaled system: rotation and translation differ by ~1e3."""
    scale = 1.0 / np.linalg.norm(J, axis=0)
    information = (J * scale).T @ (J * scale)
    return sigma_px ** 2 * (np.linalg.inv(information) * np.outer(scale, scale))


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def transfer_to_marker(sigma_reference, R_reference, t_target):
    """Covariance of the expected marker pose, given the board pose covariance.

    t_exp = R_ref c + t_ref, so a board rotation dtheta moves it by -[R_ref c]x dtheta.
    """
    A = np.eye(6)
    A[3:, :3] = -skew(R_reference @ np.asarray(t_target, float))
    return A @ sigma_reference @ A.T


def residual_sigma_px(rss_px2, n_points):
    """Noise estimated from one frame: RSS / (2N - 6)."""
    return float(np.sqrt(rss_px2 / (2 * n_points - DOF)))


def sigma_at_speed(speed, base_px, gain):
    """Corner sigma that grows with the speed of the target: sigma^2 = base^2 + (gain v)^2."""
    return np.sqrt(base_px ** 2 + (gain * np.asarray(speed, float)) ** 2)


def fit_speed_sigma(nees_values, speed, sigma_used, edges):
    """Fit (base, gain) so the median NEES of each speed bin equals the chi-square median.

    NEES scales with 1 / sigma^2, so a frame evaluated with sigma_used gives its value
    under any other sigma without recomputing the covariance.
    """
    nees_values, speed = np.asarray(nees_values, float), np.asarray(speed, float)
    bins = [(speed >= a) & (speed < b) for a, b in zip(edges[:-1], edges[1:])]
    bins = [b for b in bins if b.sum() >= 30]

    def cost(p):
        scaled = nees_values * (sigma_used / sigma_at_speed(speed, np.exp(p[0]), np.exp(p[1]))) ** 2
        return sum(np.log(np.median(scaled[b]) / CHI2_6_MEDIAN) ** 2 for b in bins)

    best = minimize(cost, np.log([sigma_used * 3, 0.1]), method="Nelder-Mead",
                    options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 2000})
    return float(np.exp(best.x[0])), float(np.exp(best.x[1]))


def nees(e, sigma):
    """e^T Sigma^-1 e, on a scaled system for the same reason as pose_covariance."""
    scale = 1.0 / np.sqrt(np.diag(sigma))
    z = e * scale
    return float(z @ np.linalg.solve(sigma * np.outer(scale, scale), z))
