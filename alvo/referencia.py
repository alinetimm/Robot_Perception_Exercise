"""Detecção do alvo com ordem de cantos garantida.

O tabuleiro tem 9 x 6 quadrados (8 x 5 cantos internos). Por ser assimétrico,
o detector devolve sempre a mesma ordem, mas essa ordem pode começar pelo lado
oposto ao referencial de geometria_alvo.json. Para não depender da versão do
OpenCV, `cantos_tabuleiro` confere a cor do quadrado externo junto ao primeiro
canto: no alvo impresso, o quadrado do canto superior esquerdo é preto. Se
estiver branco, a ordem é invertida.

Uso típico:
    from alvo.referencia import carregar_geometria, cantos_tabuleiro, cantos_marcador
    geo = carregar_geometria()
    c = cantos_tabuleiro(gray, geo)            # (N,1,2) na ordem de geo["objp_tabuleiro"]
    m = cantos_marcador(gray, geo)             # (4,1,2) TL,TR,BR,BL ou None
"""
import json
from pathlib import Path

import cv2
import numpy as np

AQUI = Path(__file__).resolve().parent


def carregar_geometria(caminho=AQUI / "geometria_alvo.json") -> dict:
    g = json.loads(Path(caminho).read_text(encoding="utf-8"))
    nx, ny = g["tabuleiro"]["cantos_internos"]
    q = g["tabuleiro"]["quadrado_mm"]
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2) * q
    g["objp_tabuleiro"] = objp
    g["objp_marcador"] = np.array(g["marcador"]["cantos_no_referencial_do_alvo_mm"], np.float32)
    g["padrao"] = (nx, ny)
    return g


def _quadrado_externo_e_preto(gray, c, nx):
    """Amostra o centro do quadrado externo diagonal ao primeiro canto."""
    p0, p_diag = c[0, 0], c[nx + 1, 0]          # canto (0,0) e canto (1,1)
    centro = p0 + 0.5 * (p0 - p_diag)            # meio do quadrado externo (-1,-1)..(0,0)
    x, y = int(round(centro[0])), int(round(centro[1]))
    h, w = gray.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return None
    viz = gray[max(0, y - 2):y + 3, max(0, x - 2):x + 3]
    ref_claro = float(np.percentile(gray, 90))
    ref_escuro = float(np.percentile(gray, 10))
    return float(viz.mean()) < (ref_claro + ref_escuro) / 2


def cantos_tabuleiro(gray, geo, refinar=True):
    nx, ny = geo["padrao"]
    ok, c = cv2.findChessboardCornersSB(gray, (nx, ny), cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_ACCURACY)
    if not ok:
        return None
    c = c.reshape(-1, 1, 2).astype(np.float32)
    preto = _quadrado_externo_e_preto(gray, c, nx)
    if preto is False:
        c = c[::-1].copy()
    if refinar:
        c = cv2.cornerSubPix(gray, c, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3))
    return c


_DETECTORES = {}


def cantos_marcador(gray, geo):
    nome = geo["marcador"]["dicionario"]
    if nome not in _DETECTORES:
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        _DETECTORES[nome] = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, nome)), params)
    corners, ids, _ = _DETECTORES[nome].detectMarkers(gray)
    if ids is None:
        return None
    for c, i in zip(corners, ids.ravel()):
        if i == geo["marcador"]["id"]:
            return c.reshape(-1, 1, 2).astype(np.float32)
    return None
