r"""LaTeX macros and the results table. Reads results/*.json, calib/camera.yaml and
alvo/geometria_alvo.json; writes paper/numbers.tex and paper/table_results.tex, so the
text never carries a typed value.

Use in the paper: \input{numbers} and \input{table_results}.
Run: .venv/bin/python scripts/06_numbers.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.covariance import CHI2_6_HIGH, CHI2_6_LOW, CHI2_6_MEDIAN  # noqa: E402

PAPER = ROOT / "paper"
VIDEO_NAMES = {"E1": "Eone", "E1_far": "Efar", "E2": "Etwo", "E3": "Ethree", "E5": "Efive"}
TABLE_LABELS = {"E1": "E1 (fit)", "E1_far": "Far", "E2": "E2", "E3": "E3", "E5": "E5"}
SIGMA_MODELS = ("fixed", "residual", "effective", "speed")
# Each bin keeps its letter in every video, even when a video has no frames in a bin.
SPEED_BINS = dict(zip(["0-0.3", "0.3-1", "1-3", "3-10", "10-1e+06"], "ABCDE"))
TILT_BINS = dict(zip(["0-15", "15-30", "30-45", "45-60", "60-90"], "ABCDE"))
K_BINS = dict(zip(["1-5", "6-15", "16-30", "31-60", "61-1000"], "ABCDE"))


def macro(name, value, digits=2):
    """One \\newcommand; LaTeX macro names take letters only."""
    if not re.fullmatch(r"[A-Za-z]+", name):
        raise ValueError(f"Invalid macro name: {name}")
    if isinstance(value, (float, np.floating)):
        text = f"{value:.{digits}f}"
    else:
        text = str(value)
    return f"\\newcommand{{\\{name}}}{{{text}}}"


def setup_macros(camera, geometry):
    board = geometry["tabuleiro"]
    marker = geometry["marcador"]
    return [macro("chiMedian", CHI2_6_MEDIAN, 2), macro("chiLow", CHI2_6_LOW, 2), macro("chiHigh", CHI2_6_HIGH, 2),
            macro("camWidth", camera["image_width"]), macro("camHeight", camera["image_height"]),
            macro("geoCornersX", board["cantos_internos"][0]), macro("geoCornersY", board["cantos_internos"][1]),
            macro("geoCorners", board["cantos_internos"][0] * board["cantos_internos"][1]),
            macro("geoSquare", float(board["quadrado_mm"]), 0), macro("geoMarker", float(marker["lado_mm"]), 0),
            macro("geoDictionary", marker["dicionario"].replace("_", r"\_"))]


def camera_macros(camera):
    K = camera["K"]
    v, t = camera["validation"], camera["target"]
    sd = camera["stability_sd_fx_fy_cx_cy_px"]
    return [macro("calibRMS", camera["rms_px"], 2), macro("calibFrames", camera["n_frames"]),
            macro("calibPhotos", camera["candidates"]),
            macro("calibFx", K[0][0], 1), macro("calibFy", K[1][1], 1), macro("calibCx", K[0][2], 1),
            macro("calibCy", K[1][2], 1), macro("calibKone", camera["dist_k1_k2_p1_p2_k3"][0], 3),
            macro("calibKtwo", camera["dist_k1_k2_p1_p2_k3"][1], 3),
            macro("calibSdFocal", max(sd[:2]), 1), macro("calibSdCentre", max(sd[2:]), 1),
            macro("calibHeldLow", v["held_out_rms_px"][0], 2), macro("calibHeldHigh", v["held_out_rms_px"][1], 2),
            macro("calibMarkerNominal", v["marker_nominal_rms_px"], 2),
            macro("calibMarkerMeasured", v["marker_measured_rms_px"], 2),
            macro("calibBend", t["max_bend_mm"], 1),
            macro("calibShiftLow", min(t["marker_shift_mm"]), 1), macro("calibShiftHigh", max(t["marker_shift_mm"]), 1),
            macro("calibSideLow", t["marker_side_mm"][0], 1), macro("calibSideHigh", t["marker_side_mm"][1], 1)]


def sigma_macros(sigma, nees):
    out = []
    for group, name in (("mjpg", "Mjpg"), ("vp8", "Vpeight")):
        out.append(macro(f"sigmaMarker{name}", sigma["groups"][group]["sigma_marker_px"], 2))
        out.append(macro(f"sigmaBoard{name}", sigma["groups"][group]["sigma_board_px"], 2))
    model = nees["speed_model"]
    return out + [macro("sigmaEffective", nees["effective"]["sigma_marker_px"], 2),
                  macro("sigmaFactor", nees["sigma_factor"], 1),
                  macro("speedBase", model["base_px"], 2), macro("speedGain", model["gain"], 2)]


def video_macros(nees):
    out = []
    for video, name in VIDEO_NAMES.items():
        entry = nees["videos"][video]
        out += [macro(f"{name}Frames", entry["detection"]["frames"]),
                macro(f"{name}Both", 100 * entry["detection"]["both"], 1),
                macro(f"{name}Rot", entry["error"]["median_rot_deg"], 2),
                macro(f"{name}Trans", entry["error"]["median_trans_mm"], 1),
                macro(f"{name}Distance", entry["error"]["median_distance_mm"] / 1000, 2),
                macro(f"{name}MaxTilt", entry["error"]["max_tilt_deg"], 0),
                macro(f"{name}Wrong", 100 * entry["error"]["second_closer"], 1)]
        for model in SIGMA_MODELS:
            label = model.capitalize()
            out += [macro(f"{name}Anees{label}", entry[model]["anees"], 1),
                    macro(f"{name}Median{label}", entry[model]["median"], 1),
                    macro(f"{name}Inside{label}", 100 * entry[model]["inside_95"], 0)]
        for key, part in entry["by_speed"].items():
            letter = SPEED_BINS[key]
            for model in ("fixed", "effective", "speed"):
                out.append(macro(f"{name}Speed{letter}{model.capitalize()}", part[model]["median"], 1))
            out.append(macro(f"{name}Speed{letter}Inside", 100 * part["speed"]["inside_95"], 0))
        for key, part in entry["by_tilt"].items():
            letter = TILT_BINS[key]
            out += [macro(f"{name}Tilt{letter}Rot", part["median_rot_deg"], 2),
                    macro(f"{name}Tilt{letter}Wrong", 100 * part["second_closer"], 1),
                    macro(f"{name}Tilt{letter}Frames", part["frames"])]
    return out


def occlusion_macros(occlusion):
    out = [macro("occWindows", occlusion["counts"]["synthetic_windows"]),
           macro("occReal", occlusion["counts"]["real_occlusions"])]
    for method in ("hold", "corners", "ring"):
        for group, name in (("E2_test", "Etwo"), ("E3", "Ethree"), ("E5", "Efive")):
            data = occlusion["methods"][method]["sets"][group]
            prefix = f"occ{method.capitalize()}{name}"
            whole = data["all"]
            out.append(macro(f"{prefix}Lost", 100 * whole["lost"], 0))
            if "median_rot_deg" in whole:
                out += [macro(f"{prefix}Rot", whole["median_rot_deg"], 2),
                        macro(f"{prefix}Trans", whole["median_trans_mm"], 1)]
            for key, part in data["by_k"].items():
                letter = K_BINS[key]
                out.append(macro(f"{prefix}K{letter}Lost", 100 * part["lost"], 0))
                if "nees_median" in part:
                    out += [macro(f"{prefix}K{letter}Rot", part["median_rot_deg"], 2),
                            macro(f"{prefix}K{letter}RotHigh", part["p95_rot_deg"], 1),
                            macro(f"{prefix}K{letter}Trans", part["median_trans_mm"], 1),
                            macro(f"{prefix}K{letter}TransHigh", part["p95_trans_mm"], 1),
                            macro(f"{prefix}K{letter}Nees", part["nees_median"], 1),
                            macro(f"{prefix}K{letter}Inside", 100 * part["inside_95"], 0)]
    return out


def results_table(nees):
    lines = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
             r" & & & \multicolumn{2}{c}{Median error} & \multicolumn{2}{c}{Fixed $\sigma$} & "
             r"\multicolumn{2}{c}{Speed $\sigma$} \\",
             r"Condition & Frames & Detected (\%) & deg & mm & ANEES & In 95\% & ANEES & In 95\% \\", r"\midrule"]
    for video, label in TABLE_LABELS.items():
        e = nees["videos"][video]
        lines.append(f"{label} & {e['detection']['frames']} & {100 * e['detection']['both']:.1f} & "
                     f"{e['error']['median_rot_deg']:.2f} & {e['error']['median_trans_mm']:.1f} & "
                     f"{e['fixed']['anees']:.0f} & {100 * e['fixed']['inside_95']:.0f} & "
                     f"{e['speed']['anees']:.1f} & {100 * e['speed']['inside_95']:.0f} \\\\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n"


def main():
    nees = json.loads((ROOT / "results/nees.json").read_text(encoding="utf-8"))
    sigma = json.loads((ROOT / "results/sigma.json").read_text(encoding="utf-8"))
    occlusion = json.loads((ROOT / "results/occlusion.json").read_text(encoding="utf-8"))
    camera = yaml.safe_load((ROOT / "calib/camera.yaml").read_text(encoding="utf-8"))
    geometry = json.loads((ROOT / "alvo/geometria_alvo.json").read_text(encoding="utf-8"))
    header = "% Generated by scripts/06_numbers.py. Do not edit.\n"
    macros = (setup_macros(camera, geometry) + camera_macros(camera) + sigma_macros(sigma, nees)
              + video_macros(nees) + occlusion_macros(occlusion))
    (PAPER / "numbers.tex").write_text(header + "\n".join(macros) + "\n", encoding="utf-8")
    (PAPER / "table_results.tex").write_text(header + results_table(nees), encoding="utf-8")
    print(f"{len(macros)} macros")


if __name__ == "__main__":
    main()
