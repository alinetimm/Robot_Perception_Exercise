"""PnP and marker error on data of known pose (CLAUDE.md: synthetic first)."""
import cv2
import numpy as np
import pytest

from alvo.referencia import carregar_geometria
from render_target import TargetRenderer
from src.geometry import exp_so3, geodesic_rad, log_so3, pose_error
from src.pose import (Camera, board_pose, expected_marker_pose, marker_model, marker_poses, process_frame,
                      tilt_deg)

K = np.array([[1420.0, 0, 632], [0, 1420, 469], [0, 0, 1]])
CAM = Camera(K, np.zeros(5))
GEO = carregar_geometria()
MODEL = marker_model(GEO["objp_marcador"])


def project(points, R, t, camera=CAM):
    return cv2.projectPoints(np.asarray(points, float), cv2.Rodrigues(R)[0], t, camera.K,
                             camera.dist)[0].astype(np.float32)


def known_pose(tilt=30.0, axis=40.0, distance=700.0):
    a = np.radians(axis)
    R = exp_so3(np.radians(tilt) * np.array([np.cos(a), np.sin(a), 0]))
    return R, np.array([-70.0, -80.0, distance])


def test_exp_log_roundtrip_and_geodesic():
    w = np.array([0.3, -0.2, 0.5])
    assert log_so3(exp_so3(w)) == pytest.approx(w)
    assert geodesic_rad(np.eye(3), exp_so3(w)) == pytest.approx(np.linalg.norm(w))


def test_pose_error_inverts_the_left_perturbation():
    R, t = known_pose()
    e = np.array([0.01, -0.02, 0.03, 1.0, -2.0, 3.0])
    assert pose_error(exp_so3(e[:3]) @ R, t + e[3:], R, t) == pytest.approx(e)


def test_marker_model_of_the_nominal_target():
    assert MODEL.t_target == pytest.approx([70, 168, 0])
    assert MODEL.R_target == pytest.approx(np.eye(3))
    assert MODEL.points[:, :2] == pytest.approx(np.array([[-40, -40], [40, -40], [40, 40], [-40, 40]]))


def test_marker_model_keeps_a_tilted_sheet():
    R_true = exp_so3([0.05, -0.03, 0.02])
    model = marker_model(MODEL.points @ R_true.T + [70, 168, 2])
    assert model.R_target == pytest.approx(R_true, abs=1e-9)
    assert model.t_target == pytest.approx([70, 168, 2])


@pytest.mark.parametrize("tilt", [0, 30, 60])
def test_board_pnp_recovers_the_pose(tilt):
    R, t = known_pose(tilt)
    pose = board_pose(project(GEO["objp_tabuleiro"], R, t), GEO["objp_tabuleiro"], CAM)
    assert geodesic_rad(pose.R, R) < 1e-3
    assert pose.t == pytest.approx(t, abs=0.05)
    assert tilt_deg(pose.R) == pytest.approx(np.degrees(np.arccos(R[2, 2])), abs=0.1)


def test_board_solution_behind_the_camera_is_mirrored_to_the_front(monkeypatch):
    """Seen on real far, frontal frames: same reprojection error, 180 degrees of rotation error."""
    R, t = known_pose(5, distance=1660)
    corners = project(GEO["objp_tabuleiro"], R, t)
    real = cv2.solvePnP

    def behind_on_first_call(points, image, K_, dist, *guess, **options):
        if guess or options:
            return real(points, image, K_, dist, *guess, **options)
        ok, rvec, tvec = real(points, image, K_, dist)
        return ok, cv2.Rodrigues(cv2.Rodrigues(rvec)[0] @ np.diag([-1.0, -1.0, 1.0]))[0], -tvec

    monkeypatch.setattr("src.pose.cv2.solvePnP", behind_on_first_call)
    pose = board_pose(corners, GEO["objp_tabuleiro"], CAM)
    assert pose.t[2] > 0
    assert geodesic_rad(pose.R, R) < 1e-3
    assert pose.t == pytest.approx(t, abs=0.1)


@pytest.mark.parametrize("tilt", [15, 40, 65])
def test_marker_ippe_recovers_the_pose_and_gives_two_solutions(tilt):
    R, t = known_pose(tilt)
    solutions = marker_poses(project(MODEL.points, R, t), MODEL, CAM)
    assert len(solutions) == 2
    assert solutions[0].rms_px <= solutions[1].rms_px
    assert geodesic_rad(solutions[0].R, R) < 1e-3
    assert solutions[0].t == pytest.approx(t, abs=0.05)
    assert geodesic_rad(solutions[1].R, R) > np.radians(5)   # the mirrored solution


def test_marker_with_distortion():
    camera = Camera(K, np.array([0.07, 0.07, 0, 0, 0.0]))
    R, t = known_pose(35)
    best = marker_poses(project(MODEL.points, R, t, camera), MODEL, camera)[0]
    assert geodesic_rad(best.R, R) < 1e-3


def test_expected_marker_pose_composition():
    R, t = known_pose()
    reference = board_pose(project(GEO["objp_tabuleiro"], R, t), GEO["objp_tabuleiro"], CAM)
    R_exp, t_exp = expected_marker_pose(reference, MODEL)
    direct = project(GEO["objp_marcador"], R, t)
    through_marker = project(MODEL.points, R_exp, t_exp)
    assert np.abs(direct - through_marker).max() < 0.01


def test_noisy_corners_bias_stays_small():
    """Mean marker error over noisy corners is near zero: no sign or frame mistake."""
    rng = np.random.default_rng(0)
    R, t = known_pose(30)
    clean = project(MODEL.points, R, t)
    errors = []
    for _ in range(400):
        noisy = (clean + rng.normal(0, 0.1, clean.shape)).astype(np.float32)
        best = marker_poses(noisy, MODEL, CAM)[0]
        errors.append(pose_error(best.R, best.t, R, t))
    errors = np.array(errors)
    sem = errors.std(0) / np.sqrt(len(errors))
    assert np.all(np.abs(errors.mean(0)) < 4 * sem + 1e-9)


@pytest.mark.parametrize("tilt,axis", [(0, 0), (25, 30), (50, 100)])
def test_process_frame_on_a_rendered_image(tilt, axis):
    renderer = TargetRenderer(K)
    R, t = renderer.pose(tilt, axis, (632, 400), 600)
    row = process_frame(renderer.render(R, t), GEO, MODEL, CAM)
    assert row["board_ok"] and row["marker_ok"]
    assert geodesic_rad(row["ref_R"], R) < np.radians(0.3)
    assert row["ref_t"] == pytest.approx(t, abs=2)
    assert np.degrees(np.linalg.norm(row["err"][:3])) < 2
    assert np.linalg.norm(row["err"][3:]) < 5


def test_process_frame_without_a_target():
    row = process_frame(np.full((960, 1280), 128, np.uint8), GEO, MODEL, CAM)
    assert not row["board_ok"] and not row["marker_ok"] and "err" not in row


def test_marker_speed_uses_only_the_previous_frame():
    import pandas as pd

    from src.frames import marker_speed
    columns = {f"mc{i}_{a}": [0.0, 3.0, 3.0, 9.0, 12.0] for i in range(4) for a in "x"}
    columns.update({f"mc{i}_y": [0.0] * 5 for i in range(4)})
    df = pd.DataFrame({"frame": [0, 1, 2, 4, 5], "time_s": np.array([0, 1, 2, 4, 5]) / 30, **columns})
    speed = marker_speed(df)
    assert speed[1] == pytest.approx(3) and speed[2] == pytest.approx(0)
    assert speed[0] == pytest.approx(3)    # no past frame: next known value
    assert speed[3] == pytest.approx(3)    # gap in the frames: next known value
