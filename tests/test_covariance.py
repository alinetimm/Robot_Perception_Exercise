"""Covariance and NEES against Monte Carlo with known corner noise (CLAUDE.md: ANEES near 6)."""
import cv2
import numpy as np
import pytest

from alvo.referencia import carregar_geometria
from src.covariance import (CHI2_6_HIGH, CHI2_6_LOW, CHI2_6_MEDIAN, fit_speed_sigma, jacobian, sigma_at_speed, nees, pose_covariance, project, residual_sigma_px,
                            skew, transfer_to_marker)
from src.geometry import exp_so3, pose_error
from src.pose import Camera, board_pose, expected_marker_pose, marker_model, marker_poses

K = np.array([[1420.0, 0, 632], [0, 1420, 469], [0, 0, 1]])
CAM = Camera(K, np.array([0.07, 0.07, 0, 0, 0.0]))
GEO = carregar_geometria()
MODEL = marker_model(GEO["objp_marcador"])
BOARD = np.asarray(GEO["objp_tabuleiro"], float)
SIGMA = 0.1


def poses(tilt, distance=600.0):
    a = np.radians(35)
    R = exp_so3(np.radians(tilt) * np.array([np.cos(a), np.sin(a), 0]))
    return R, np.array([-60.0, -130.0, distance])


def test_jacobian_predicts_a_small_perturbation():
    R, t = poses(30)
    J = jacobian(MODEL.points, R, t, CAM)
    d = np.array([1e-4, -2e-4, 1.5e-4, 0.02, -0.03, 0.05])
    moved = project(MODEL.points, exp_so3(d[:3]) @ R, t + d[3:], CAM)
    assert moved - project(MODEL.points, R, t, CAM) == pytest.approx(J @ d, abs=5e-5)   # second-order term


def test_jacobian_near_the_identity_rotation():
    """cv2.Rodrigues returns zero below 1e-5 rad, which once zeroed these columns."""
    J = jacobian(np.array([[0.0, 0.0, -100.0]]), np.eye(3), np.array([0.0, 0.0, 700.0]), CAM)
    assert np.abs(J[:, :2]).max() > 100


def test_chi2_interval_constants():
    scipy_stats = pytest.importorskip("scipy.stats")
    assert scipy_stats.chi2.ppf([0.025, 0.975], 6) == pytest.approx([CHI2_6_LOW, CHI2_6_HIGH], abs=1e-3)
    assert scipy_stats.chi2.median(6) == pytest.approx(CHI2_6_MEDIAN, abs=1e-3)


def test_nees_of_a_known_vector():
    sigma = np.diag([1e-6, 4e-6, 1e-6, 1.0, 4.0, 9.0])
    e = np.array([1e-3, 2e-3, 0, 1, 2, 3])
    assert nees(e, sigma) == pytest.approx(5.0)


def test_transfer_matches_a_direct_perturbation():
    """The linear map equals moving the board pose and recomputing the expected marker pose."""
    R, t = poses(30)
    reference = type("P", (), {"R": R, "t": t})
    R0, t0 = expected_marker_pose(reference, MODEL)
    d = np.array([1e-5, -2e-5, 3e-5, 0.01, 0.02, -0.01])
    moved = type("P", (), {"R": exp_so3(d[:3]) @ R, "t": t + d[3:]})
    got = pose_error(*expected_marker_pose(moved, MODEL), R0, t0)
    A = transfer_to_marker(np.eye(6), R, MODEL.t_target)   # A A^T; recover A through skew
    A_direct = np.eye(6)
    A_direct[3:, :3] = -skew(R @ MODEL.t_target)
    assert A == pytest.approx(A_direct @ A_direct.T)
    assert got == pytest.approx(A_direct @ d, abs=1e-6)


def monte_carlo(tilt, runs=3000, seed=0):
    rng = np.random.default_rng(seed)
    R, t = poses(tilt)
    clean = cv2.projectPoints(MODEL.points, cv2.Rodrigues(R)[0], t, K, CAM.dist)[0]
    errors, values, sigmas = [], [], []
    for _ in range(runs):
        noisy = (clean + rng.normal(0, SIGMA, clean.shape)).astype(np.float32)
        best = marker_poses(noisy, MODEL, CAM)[0]
        cov = pose_covariance(jacobian(MODEL.points, best.R, best.t, CAM), SIGMA)
        e = pose_error(best.R, best.t, R, t)
        errors.append(e)
        values.append(nees(e, cov))
        sigmas.append(np.sqrt(2) * best.rms_px)   # RSS = 4 rms^2, RSS / 2 = 2 rms^2
    return np.array(errors), np.array(values), np.array(sigmas)


@pytest.mark.parametrize("tilt", [30, 45])
def test_anees_near_6_with_known_noise(tilt):
    _, values, _ = monte_carlo(tilt)
    assert values.mean() == pytest.approx(6, abs=0.4)
    inside = ((values >= CHI2_6_LOW) & (values <= CHI2_6_HIGH)).mean()
    assert inside == pytest.approx(0.95, abs=0.02)


def test_predicted_covariance_matches_the_sample_covariance():
    errors, _, _ = monte_carlo(40)
    R, t = poses(40)
    predicted = pose_covariance(jacobian(MODEL.points, R, t, CAM), SIGMA)
    ratio = np.diag(np.cov(errors.T)) / np.diag(predicted)
    assert ratio == pytest.approx(np.ones(6), abs=0.15)


def test_residual_sigma_is_unbiased_in_variance_but_unstable():
    _, _, sigmas = monte_carlo(40)
    assert np.mean(sigmas ** 2) == pytest.approx(SIGMA ** 2, rel=0.1)
    assert sigmas.std() / sigmas.mean() > 0.4   # 2 degrees of freedom only
    assert residual_sigma_px(8.0, 4) == pytest.approx(2.0)


def test_anees_near_6_with_noisy_reference_and_marker():
    """Full chain: both poses are estimated, and the error uses the summed covariance."""
    rng = np.random.default_rng(1)
    R, t = poses(35)
    board_clean = cv2.projectPoints(BOARD, cv2.Rodrigues(R)[0], t, K, CAM.dist)[0]
    reference_true = type("P", (), {"R": R, "t": t})
    R_m, t_m = expected_marker_pose(reference_true, MODEL)
    marker_clean = cv2.projectPoints(MODEL.points, cv2.Rodrigues(R_m)[0], t_m, K, CAM.dist)[0]
    values = []
    for _ in range(2000):
        board = (board_clean + rng.normal(0, SIGMA, board_clean.shape)).astype(np.float32)
        marker = (marker_clean + rng.normal(0, SIGMA, marker_clean.shape)).astype(np.float32)
        reference = board_pose(board, BOARD, CAM)
        best = marker_poses(marker, MODEL, CAM)[0]
        R_exp, t_exp = expected_marker_pose(reference, MODEL)
        total = (pose_covariance(jacobian(MODEL.points, best.R, best.t, CAM), SIGMA)
                 + transfer_to_marker(pose_covariance(jacobian(BOARD, reference.R, reference.t, CAM), SIGMA),
                                      reference.R, MODEL.t_target))
        values.append(nees(pose_error(best.R, best.t, R_exp, t_exp), total))
    assert np.mean(values) == pytest.approx(6, abs=0.4)


def test_speed_fit_recovers_a_known_noise_law():
    rng = np.random.default_rng(3)
    speed = np.exp(rng.uniform(np.log(0.05), np.log(30), 40000))
    true = sigma_at_speed(speed, 0.4, 0.25)
    values = rng.chisquare(6, len(speed)) * (true / 0.13) ** 2   # NEES under a too small sigma
    base, gain = fit_speed_sigma(values, speed, 0.13, [0, 0.3, 1, 3, 10, 1e6])
    assert base == pytest.approx(0.4, rel=0.1)
    assert gain == pytest.approx(0.25, rel=0.1)


def test_underestimated_sigma_shows_up_in_the_nees():
    """Sanity check of the metric: halving sigma must multiply the NEES by four."""
    errors, values, _ = monte_carlo(40, runs=300)
    R, t = poses(40)
    cov = pose_covariance(jacobian(MODEL.points, R, t, CAM), SIGMA / 2)
    assert np.mean([nees(e, cov) for e in errors]) == pytest.approx(4 * values.mean(), rel=0.1)
