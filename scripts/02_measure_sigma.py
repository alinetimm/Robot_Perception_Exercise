"""Corner noise sigma, per video compression, from the still stretches of the videos.

Reads data/proc/frames/, writes results/sigma.json.

Hand-held or vibrating targets are fine: each corner residual is taken against the
prediction from the same frame, so a rigid motion of the sheet cancels.

The sigma used by the NEES comes from the difference of the residual between
consecutive frames of a whole video group: a fixed model error cancels in the
difference, and a robust scale (MAD) ignores the frames of fast motion. Still stretches
give a second estimate, reported for comparison. Marker residuals include the small
noise of the board pose, so sigma_marker is slightly overestimated. Inter-frame video
compression can correlate consecutive frames, which would lower the VP8 value.

Run: .venv/bin/python scripts/02_measure_sigma.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alvo.referencia import carregar_geometria_medida  # noqa: E402
from src.covariance import DOF, project  # noqa: E402
from src.frames import corners, load, pose  # noqa: E402
from src.pose import expected_marker_pose, load_camera, marker_model  # noqa: E402

GROUPS = {"mjpg": ["E1", "E1_far"], "vp8": ["E1_webm", "E2", "E3", "E5"]}
MAX_DRIFT_PX = 3.0
MIN_STRETCH_S = 2.0
MIN_FRAMES = 20
N_BOARD = 40


def still_stretches(df):
    """Maximal runs of frames whose board stays within MAX_DRIFT_PX of the run's first frame."""
    board = corners(df, "bc", N_BOARD)
    usable = (df.board_ok & df.marker_ok).to_numpy()
    time = df.time_s.to_numpy()
    runs, start = [], 0
    while start < len(df):
        if not usable[start]:
            start += 1
            continue
        stop = start
        while (stop + 1 < len(df) and usable[stop + 1]
               and np.linalg.norm(board[stop + 1] - board[start], axis=1).mean() <= MAX_DRIFT_PX):
            stop += 1
        if time[stop] - time[start] >= MIN_STRETCH_S and stop - start + 1 >= MIN_FRAMES:
            runs.append((start, stop))
            start = stop + 1
        else:
            start += 1
    return runs


def residuals(df, geo, model, camera):
    """Corner residuals of the marker and of the board, for each frame of df."""
    marker_res, board_res = [], []
    board_seen, marker_seen = corners(df, "bc", N_BOARD), corners(df, "mc", 4)
    board_points = np.asarray(geo["objp_tabuleiro"], float)
    for i, (_, row) in enumerate(df.iterrows()):
        R, t = pose(row, "ref")
        reference = type("Pose", (), {"R": R, "t": t})
        R_exp, t_exp = expected_marker_pose(reference, model)
        marker_res.append(marker_seen[i] - project(model.points, R_exp, t_exp, camera).reshape(-1, 2))
        board_res.append(board_seen[i] - project(board_points, R, t, camera).reshape(-1, 2))
    return np.array(marker_res), np.array(board_res)


def difference_scale(steps):
    """Robust noise from frame-to-frame differences of the residual, per coordinate."""
    return float(1.4826 * np.median(np.abs(np.concatenate(steps))) / np.sqrt(2))


def stretch_noise(df, geo, model, camera):
    """Sums of squares and degrees of freedom of the corner noise in one stretch."""
    marker_res, board_res = residuals(df, geo, model, camera)
    n = len(df)
    marker_ss = float(((marker_res - marker_res.mean(0)) ** 2).sum())
    board_ss = float(((board_res - board_res.mean(0)) ** 2).sum())
    return marker_ss, (n - 1) * 8, board_ss, (n - 1) * (2 * N_BOARD - DOF)


def main():
    geo = carregar_geometria_medida()
    model = marker_model(geo["objp_marcador"])
    camera = load_camera(ROOT / "calib/camera.yaml")
    result = {"max_drift_px": MAX_DRIFT_PX, "min_stretch_s": MIN_STRETCH_S, "groups": {}, "stretches": []}
    for group, videos in GROUPS.items():
        total = np.zeros(4)
        marker_steps, board_steps = [], []
        for name in videos:
            df = load(name)
            both = df[df.board_ok & df.marker_ok]
            marker_res, board_res = residuals(both, geo, model, camera)
            consecutive = np.diff(both.frame.to_numpy()) == 1
            marker_steps.append((marker_res[1:] - marker_res[:-1])[consecutive].ravel())
            board_steps.append((board_res[1:] - board_res[:-1])[consecutive].ravel())
            for start, stop in still_stretches(df):
                part = df.iloc[start:stop + 1]
                noise = np.array(stretch_noise(part, geo, model, camera))
                total += noise
                result["stretches"].append({
                    "group": group, "video": name, "start_s": float(part.time_s.iloc[0]),
                    "duration_s": float(part.time_s.iloc[-1] - part.time_s.iloc[0]), "frames": len(part),
                    "tilt_deg": float(part.tilt_deg.median()), "distance_mm": float(part.distance_mm.median()),
                    "sigma_marker_px": float(np.sqrt(noise[0] / noise[1])),
                    "sigma_board_px": float(np.sqrt(noise[2] / noise[3]))})
        rows = [s for s in result["stretches"] if s["group"] == group]
        result["groups"][group] = {
            "sigma_marker_px": difference_scale(marker_steps),
            "sigma_board_px": difference_scale(board_steps),
            "frames": int(sum(len(s) for s in marker_steps) // 8),
            "stretch_sigma_marker_px": float(np.sqrt(total[0] / total[1])) if total[1] else None,
            "stretch_sigma_board_px": float(np.sqrt(total[2] / total[3])) if total[3] else None,
            "stretches": len(rows), "stretch_frames": sum(s["frames"] for s in rows)}
        print(group, result["groups"][group], flush=True)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/sigma.json").write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
