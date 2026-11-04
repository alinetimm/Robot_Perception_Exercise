"""Smoke test: detect the chessboard and the marker on the target preview."""
from pathlib import Path

import cv2

from alvo.referencia import cantos_marcador, cantos_tabuleiro, carregar_geometria

PREVIEW = Path(__file__).resolve().parents[1] / "alvo" / "alvo_A4_preview.png"


def _load():
    gray = cv2.imread(str(PREVIEW), cv2.IMREAD_GRAYSCALE)
    assert gray is not None, f"could not read {PREVIEW}"
    return gray, carregar_geometria()


def test_detects_40_chessboard_corners():
    gray, geo = _load()
    corners = cantos_tabuleiro(gray, geo)
    assert corners is not None
    assert corners.shape == (40, 1, 2)


def test_detects_marker_id_0():
    gray, geo = _load()
    assert geo["marcador"]["id"] == 0
    corners = cantos_marcador(gray, geo)
    assert corners is not None
    assert corners.shape == (4, 1, 2)
