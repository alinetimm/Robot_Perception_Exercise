"""Rules of CLAUDE.md for the paper source: generated numbers, verified citations, plain style."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "paper/main.tex"
NUMBERS = ROOT / "paper/numbers.tex"
RESULT_MACRO = re.compile(r"\\((?:calib|sigma|speed|geo|cam|chi|occ|E(?:one|two|three|five|far))[A-Z][A-Za-z]*)")
BANNED = ["crucial", "critical", "novel", "remarkable", "robust", "significant", "moreover", "furthermore",
          "in conclusion", "it is worth noting", "delve", "seamlessly", "while"]


@pytest.fixture(scope="module")
def text():
    return MAIN.read_text(encoding="utf-8")


def prose(text):
    """The paper without comments, so notes to the author do not count."""
    return "\n".join(line.split("%")[0] if not line.lstrip().startswith("%") else "" for line in text.splitlines())


def test_every_result_macro_is_generated(text):
    if not NUMBERS.exists():
        pytest.skip("run scripts/06_numbers.py first")
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", NUMBERS.read_text(encoding="utf-8")))
    used = set(RESULT_MACRO.findall(prose(text)))
    assert used, "the paper uses no generated numbers"
    assert used <= defined, f"undefined: {sorted(used - defined)}"


def test_no_dashes_as_punctuation(text):
    assert "\u2014" not in text and "\u2013" not in text
    assert not re.search(r"[A-Za-z]---?[A-Za-z ]", prose(text).replace("\\,", ""))


def test_no_banned_words(text):
    lowered = prose(text).lower()
    found = [w for w in BANNED if re.search(rf"\b{w}\b", lowered)]
    assert not found, found


def test_citations_exist_in_the_bibliography(text):
    keys = set(re.findall(r"@\w+\{([^,]+),", (ROOT / "paper/refs.bib").read_text(encoding="utf-8")))
    cited = {k.strip() for group in re.findall(r"\\cite\{([^}]+)\}", text) for k in group.split(",")}
    assert cited and cited <= keys, sorted(cited - keys)


def test_every_figure_exists(text):
    for name in re.findall(r"\\includegraphics\[[^]]*\]\{([^}]+)\}", text):
        assert (ROOT / "figures" / f"{name}.pdf").exists(), name


def test_every_reference_has_a_doi():
    entries = re.split(r"\n@", (ROOT / "paper/refs.bib").read_text(encoding="utf-8"))[1:]
    assert entries and all(re.search(r"doi\s*=\s*\{10\.", e) for e in entries)
