"""Per-frame covariance and NEES of the marker pose against the board reference.

Reads data/proc/frames/ and results/sigma.json. Writes results/nees.json, and the
per-frame values for the figures to data/proc/nees/.

Four noise models. The first two come from PLANO.md: (a) fixed sigma, the corner
noise between frames, and (b) residual sigma from the frame itself (RSS / 2). The
real error also has a fixed part per pose that (a) cannot see, so (c) effective sigma
scales (a) until the median NEES of the fit video equals the chi-square median, and
(d) speed sigma, sigma^2 = base^2 + (gain v)^2, also grows with the marker speed v.
(c) and (d) are fitted on E1 only; every other video is an independent check. The
board covariance is included in all four.

Run: .venv/bin/python scripts/03_nees.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alvo.referencia import carregar_geometria_medida  # noqa: E402
from src.covariance import (CHI2_6_HIGH, CHI2_6_LOW, CHI2_6_MEDIAN, fit_speed_sigma, jacobian, nees,  # noqa: E402
                            pose_covariance, residual_sigma_px, sigma_at_speed, transfer_to_marker)
from src.frames import load, marker_speed, pose  # noqa: E402
from src.geometry import pose_error  # noqa: E402
from src.pose import expected_marker_pose, load_camera, marker_model  # noqa: E402

VIDEOS = {"E1": "mjpg", "E1_far": "mjpg", "E2": "vp8", "E3": "vp8", "E5": "vp8"}
TILT_BINS = [0, 15, 30, 45, 60, 90]
SPEED_BINS = [0, 0.3, 1, 3, 10, 1e6]
FIT_VIDEO = "E1"
OUT = ROOT / "data/proc/nees"


def frame_nees(row, board_points, model, camera, sigma, effective, speed_model):
    R_ref, t_ref = pose(row, "ref")
    R_m, t_m = pose(row, "m")
    reference = type("Pose", (), {"R": R_ref, "t": t_ref})
    R_exp, t_exp = expected_marker_pose(reference, model)
    e = pose_error(R_m, t_m, R_exp, t_exp)
    J = jacobian(model.points, R_m, t_m, camera)
    unit_marker = pose_covariance(J, 1.0)
    unit_reference = transfer_to_marker(pose_covariance(jacobian(board_points, R_ref, t_ref, camera), 1.0),
                                        R_ref, model.t_target)
    cov_reference = sigma["sigma_board_px"] ** 2 * unit_reference
    cov_fixed = sigma["sigma_marker_px"] ** 2 * unit_marker + cov_reference
    cov_effective = (effective["sigma_marker_px"] ** 2 * unit_marker
                     + effective["sigma_board_px"] ** 2 * unit_reference)
    sigma_speed = float(sigma_at_speed(row["speed"], speed_model["base_px"], speed_model["gain"]))
    cov_speed = sigma_speed ** 2 * (unit_marker + speed_model["board_ratio"] ** 2 * unit_reference)
    sigma_residual = residual_sigma_px(4 * row["m_rms_px"] ** 2, 4)
    cov_residual = max(sigma_residual, 1e-6) ** 2 * unit_marker + cov_reference
    out = {"frame": row["frame"], "time_s": row["time_s"], "tilt_deg": row["tilt_deg"],
           "distance_mm": row["distance_mm"], "err_rot_deg": row["err_rot_deg"], "err_trans_mm": row["err_trans_mm"],
           "ippe_ratio": row["ippe_ratio"], "second_closer": row["err2_rot_deg"] < row["err_rot_deg"],
           "speed": row["speed"], "sigma_residual_px": sigma_residual, "sigma_speed_px": sigma_speed,
           "nees_fixed": nees(e, cov_fixed), "nees_residual": nees(e, cov_residual),
           "nees_effective": nees(e, cov_effective), "nees_speed": nees(e, cov_speed),
           "nees_rot": nees(e[:3], cov_fixed[:3, :3]), "nees_trans": nees(e[3:], cov_fixed[3:, 3:]),
           "reference_share": float(np.trace(cov_reference[3:, 3:]) / np.trace(cov_fixed[3:, 3:]))}
    for i, axis in enumerate("xyz"):
        out[f"e_rot_{axis}_deg"] = float(np.degrees(e[i]))
        out[f"e_t_{axis}_mm"] = e[3 + i]
        out[f"sd_rot_{axis}_deg"] = float(np.degrees(np.sqrt(cov_fixed[i, i])))
        out[f"sd_t_{axis}_mm"] = float(np.sqrt(cov_fixed[3 + i, 3 + i]))
        out[f"sd_eff_rot_{axis}_deg"] = float(np.degrees(np.sqrt(cov_effective[i, i])))
        out[f"sd_eff_t_{axis}_mm"] = float(np.sqrt(cov_effective[3 + i, 3 + i]))
        out[f"sd_speed_rot_{axis}_deg"] = float(np.degrees(np.sqrt(cov_speed[i, i])))
        out[f"sd_speed_t_{axis}_mm"] = float(np.sqrt(cov_speed[3 + i, 3 + i]))
    return out


def summarise(values):
    values = values[np.isfinite(values)]
    if not len(values):
        return {"frames": 0}
    return {"frames": int(len(values)), "anees": float(values.mean()), "median": float(np.median(values)),
            "inside_95": float(((values >= CHI2_6_LOW) & (values <= CHI2_6_HIGH)).mean())}


def main():
    geo = carregar_geometria_medida()
    model = marker_model(geo["objp_marcador"])
    camera = load_camera(ROOT / "calib/camera.yaml")
    board_points = np.asarray(geo["objp_tabuleiro"], float)
    sigmas = json.loads((ROOT / "results/sigma.json").read_text(encoding="utf-8"))["groups"]
    OUT.mkdir(parents=True, exist_ok=True)

    fit_group = sigmas[VIDEOS[FIT_VIDEO]]
    board_ratio = fit_group["sigma_board_px"] / fit_group["sigma_marker_px"]

    def table_of(name, effective, speed_model):
        df = load(name)
        both = df[df.board_ok & df.marker_ok].copy()
        marked = df[df.marker_ok].copy()
        marked["speed"] = marker_speed(marked)
        both["speed"] = marked.speed.reindex(both.index)
        group = sigmas[VIDEOS[name]]
        return pd.DataFrame([frame_nees(r, board_points, model, camera, group, effective, speed_model)
                             for r in both.to_dict("records")])

    # Every covariance scales with sigma^2, so both fits work on the NEES of model (a).
    first = table_of(FIT_VIDEO, fit_group, {"base_px": 1.0, "gain": 0.0, "board_ratio": board_ratio})
    factor = float(np.sqrt(first.nees_fixed.median() / CHI2_6_MEDIAN))
    effective = {"sigma_marker_px": factor * fit_group["sigma_marker_px"],
                 "sigma_board_px": factor * fit_group["sigma_board_px"]}
    base, gain = fit_speed_sigma(first.nees_fixed, first.speed, fit_group["sigma_marker_px"], SPEED_BINS)
    speed_model = {"base_px": base, "gain": gain, "board_ratio": board_ratio}
    print(f"Effective sigma: x{factor:.2f}, {effective}; speed sigma: {speed_model}", flush=True)
    result = {"chi2_6_interval_95": [CHI2_6_LOW, CHI2_6_HIGH], "fit_video": FIT_VIDEO, "sigma_factor": factor,
              "effective": effective, "speed_model": speed_model, "videos": {}}
    for name, group in VIDEOS.items():
        table = table_of(name, effective, speed_model)
        table.to_parquet(OUT / f"{name}.parquet", index=False)
        frames = load(name)
        entry = {"sigma_group": group,
                 "detection": {"frames": int(len(frames)), "board": float(frames.board_ok.mean()),
                               "marker": float(frames.marker_ok.mean()),
                               "both": float((frames.board_ok & frames.marker_ok).mean())},
                 "error": {"median_rot_deg": float(table.err_rot_deg.median()),
                           "median_trans_mm": float(table.err_trans_mm.median()),
                           "second_closer": float(table.second_closer.mean())},
                 "fixed": summarise(table.nees_fixed.to_numpy()),
                 "residual": summarise(table.nees_residual.to_numpy()),
                 "effective": summarise(table.nees_effective.to_numpy()),
                 "speed": summarise(table.nees_speed.to_numpy()), "by_tilt": {}, "by_speed": {}}
        bins = pd.cut(table.tilt_deg, TILT_BINS)
        for interval, part in table.groupby(bins, observed=True):
            entry["by_tilt"][f"{interval.left:.0f}-{interval.right:.0f}"] = {
                **summarise(part.nees_fixed.to_numpy()), "effective": summarise(part.nees_effective.to_numpy()),
                "second_closer": float(part.second_closer.mean()),
                "median_rot_deg": float(part.err_rot_deg.median()), "median_trans_mm": float(part.err_trans_mm.median())}
        for interval, part in table.groupby(pd.cut(table.speed, SPEED_BINS, right=False), observed=True):
            entry["by_speed"][f"{interval.left:g}-{interval.right:g}"] = {
                "fixed": summarise(part.nees_fixed.to_numpy()), "effective": summarise(part.nees_effective.to_numpy()),
                "speed": summarise(part.nees_speed.to_numpy()),
                "median_rot_deg": float(part.err_rot_deg.median()), "median_trans_mm": float(part.err_trans_mm.median())}
        result["videos"][name] = entry
        print(name, "fixed", entry["fixed"], "effective", entry["effective"], "speed", entry["speed"],
              "residual ANEES", round(entry["residual"]["anees"], 1), flush=True)
    (ROOT / "results/nees.json").write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
