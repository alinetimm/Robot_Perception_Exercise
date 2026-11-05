"""Smoke test: detect the chessboard and the marker on the target preview."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from alvo.referencia import _refinar_pelas_bordas, cantos_marcador, cantos_tabuleiro, carregar_geometria
from render_target import BOARD_CENTER, TargetRenderer

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


@pytest.mark.parametrize("tilt,axis", [(0, 0), (30, 40), (55, 120)])
def test_edge_refinement_finds_the_true_marker_corners(tilt, axis):
    renderer = TargetRenderer(np.array([[1400.0, 0, 640], [0, 1400, 480], [0, 0, 1]]))
    R, t = renderer.pose(tilt, axis, (640, 380), 650)
    image = renderer.render(R, t)
    expected, _ = cv2.projectPoints(renderer.geo["objp_marcador"], cv2.Rodrigues(R)[0], t, renderer.K, None)
    # One pixel of the preview page is about 0.4 px here, so this image cannot rank the
    # methods below that level. The ranking comes from the real photos (calib/relatorio.md).
    error = np.linalg.norm(cantos_marcador(image, renderer.geo) - expected, axis=2)
    assert error.max() < 0.6


def test_marker_covered_at_an_edge_keeps_the_subpix_corners():
    """A hand over one edge must not break the detection."""
    renderer = TargetRenderer(np.array([[1400.0, 0, 640], [0, 1400, 480], [0, 0, 1]]))
    image = renderer.render(*renderer.pose(10, 0, (640, 380), 650))
    corners = cantos_marcador(image, renderer.geo, refinar_bordas=False)
    flat = np.full_like(image, 128)
    assert _refinar_pelas_bordas(flat, corners) is None


def _rotated_about_normal(renderer, tilt, axis, center):
    """Pose of the board turned upside down in its own plane."""
    R, _ = renderer.pose(tilt, axis, center)
    R = R @ cv2.Rodrigues(np.array([0.0, 0.0, np.pi]))[0]
    ray = np.linalg.inv(renderer.K) @ np.array([*center, 1.0])
    return R, 500 * ray - R @ BOARD_CENTER


@pytest.mark.parametrize("upside_down", [False, True])
@pytest.mark.parametrize("tilt,axis", [(0, 0), (35, 30), (60, 100)])
def test_corner_order_survives_uneven_light(upside_down, tilt, axis):
    """The old global threshold flipped the order under a light gradient."""
    renderer = TargetRenderer(np.array([[1400.0, 0, 640], [0, 1400, 480], [0, 0, 1]]))
    center = (640, 480)
    R, t = _rotated_about_normal(renderer, tilt, axis, center) if upside_down else renderer.pose(tilt, axis, center)
    image = renderer.render(R, t).astype(np.float32)
    ramp = np.linspace(0.45, 1.0, image.shape[1], dtype=np.float32)
    image = np.clip(image * ramp + 70, 0, 255).astype(np.uint8)   # washed-out blacks, shaded side
    corners = cantos_tabuleiro(image, renderer.geo)
    expected, _ = cv2.projectPoints(renderer.geo["objp_tabuleiro"], cv2.Rodrigues(R)[0], t, renderer.K, None)
    assert np.abs(corners - expected).max() < 2
