r"""LaTeX macros and the results table. Reads results/*.json, writes paper/numbers.tex
and paper/table_results.tex, so the text never carries a typed value.

Use in the paper: \input{numbers} and \input{table_results}.
Run: .venv/bin/python scripts/06_numbers.py
"""
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
VIDEO_NAMES = {"E1": "Eone", "E1_far": "Efar", "E2": "Etwo", "E3": "Ethree", "E5": "Efive"}
TABLE_LABELS = {"E1": "E1 (fit)", "E1_far": "Far, 1.66 m", "E2": "E2", "E3": "E3", "E5": "E5"}
SIGMA_MODELS = ("fixed", "residual", "effective", "speed")


def macro(name, value, digits=2):
    """One \\newcommand; LaTeX macro names take letters only."""
    if not re.fullmatch(r"[A-Za-z]+", name):
        raise ValueError(f"Invalid macro name: {name}")
    text = f"{value:.{digits}f}" if isinstance(value, float) else str(value)
    return f"\\newcommand{{\\{name}}}{{{text}}}"


def word(text):
    """Macro-safe CamelCase from a key such as '1-5'."""
    digits = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six",
              "7": "seven", "8": "eight", "9": "nine"}
    return "".join(digits.get(c, c.capitalize() if c.isalpha() else "") for c in text).replace("-", "")


def camera_macros(camera):
    K = camera["K"]
    return [macro("calibRMS", camera["rms_px"], 2), macro("calibFrames", camera["n_frames"]),
            macro("calibFx", K[0][0], 1), macro("calibFy", K[1][1], 1), macro("calibCx", K[0][2], 1),
            macro("calibCy", K[1][2], 1), macro("calibKone", camera["dist_k1_k2_p1_p2_k3"][0], 3),
            macro("calibKtwo", camera["dist_k1_k2_p1_p2_k3"][1], 3)]


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
                macro(f"{name}Wrong", 100 * entry["error"]["second_closer"], 1)]
        for model in SIGMA_MODELS:
            label = model.capitalize()
            out += [macro(f"{name}Anees{label}", entry[model]["anees"], 1),
                    macro(f"{name}Median{label}", entry[model]["median"], 1),
                    macro(f"{name}Inside{label}", 100 * entry[model]["inside_95"], 0)]
    return out


def occlusion_macros(occlusion):
    out = []
    for method in ("hold", "ring"):
        for group, name in (("E2_test", "Etwo"), ("E3", "Ethree"), ("E5", "Efive")):
            data = occlusion["methods"][method]["sets"][group]
            prefix = f"occ{method.capitalize()}{name}"
            out += [macro(f"{prefix}Rot", data["all"]["median_rot_deg"], 2),
                    macro(f"{prefix}Trans", data["all"]["median_trans_mm"], 1),
                    macro(f"{prefix}Lost", 100 * data["all"]["lost"], 0)]
            for k, v in data["by_k"].items():
                if "nees_median" in v:
                    out.append(macro(f"{prefix}Nees{word(k)}", v["nees_median"], 1))
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
    header = "% Generated by scripts/06_numbers.py. Do not edit.\n"
    macros = camera_macros(camera) + sigma_macros(sigma, nees) + video_macros(nees) + occlusion_macros(occlusion)
    (PAPER / "numbers.tex").write_text(header + "\n".join(macros) + "\n", encoding="utf-8")
    (PAPER / "table_results.tex").write_text(header + results_table(nees), encoding="utf-8")
    print(f"{len(macros)} macros")


if __name__ == "__main__":
    main()
