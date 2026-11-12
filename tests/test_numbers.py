"""Generated LaTeX macros: valid names, no duplicates, values taken from the results."""
import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("numbers_script", ROOT / "scripts" / "06_numbers.py")
numbers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(numbers)


def test_macro_format():
    assert numbers.macro("calibRMS", 0.2213, 2) == r"\newcommand{\calibRMS}{0.22}"
    assert numbers.macro("calibFrames", 156) == r"\newcommand{\calibFrames}{156}"


def test_macro_rejects_names_latex_cannot_use():
    with pytest.raises(ValueError):
        numbers.macro("E2Anees", 1.0)


def test_generated_file_matches_results():
    path = ROOT / "paper/numbers.tex"
    if not path.exists():
        pytest.skip("run scripts/06_numbers.py first")
    text = path.read_text(encoding="utf-8")
    names = re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", text)
    assert len(names) == len(set(names))
    nees = json.loads((ROOT / "results/nees.json").read_text(encoding="utf-8"))
    expected = f"{nees['videos']['E2']['speed']['anees']:.1f}"
    assert rf"\newcommand{{\EtwoAneesSpeed}}{{{expected}}}" in text


def test_bin_letters_follow_the_bin_not_its_position():
    nees = {"videos": {v: {"detection": {"frames": 1, "both": 1.0}, "error": {"median_rot_deg": 1.0,
            "median_trans_mm": 1.0, "second_closer": 0.0, "median_distance_mm": 600.0, "max_tilt_deg": 40.0},
            **{m: {"anees": 6.0, "median": 5.0, "inside_95": 0.9} for m in numbers.SIGMA_MODELS},
            "by_speed": {}, "by_tilt": {"15-30": {"median_rot_deg": 0.3, "second_closer": 0.0, "frames": 9}}}
            for v in numbers.VIDEO_NAMES}}
    text = "\n".join(numbers.video_macros(nees))
    assert r"\newcommand{\EfiveTiltBRot}{0.30}" in text
    assert "EfiveTiltARot" not in text
