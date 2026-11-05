"""Detecção do alvo com ordem de cantos garantida.

O tabuleiro tem 9 x 6 quadrados (8 x 5 cantos internos). O detector pode devolver
a ordem começando pelo lado oposto ao referencial de geometria_alvo.json. Com um
número ímpar de quadrados somados nos dois eixos, girar o tabuleiro 180° inverte a
cor de todas as casas. `_ordem_invertida` compara as casas internas entre si, sem
limiar global: elas estão sempre na imagem e não dependem da luz da cena.

Uso típico:
    from alvo.referencia import carregar_geometria, cantos_tabuleiro, cantos_marcador
    geo = carregar_geometria()                 # carregar_geometria_medida() para estimar pose
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


def carregar_geometria_medida(caminho=AQUI.parent / "calib" / "alvo_medido.json") -> dict:
    """Geometria nominal com os pontos 3D medidos por calib/calibrate.py --fotos.

    Use esta versão para pose: a folha não é perfeitamente plana. Se a folha for
    descolada ou a placa trocada, a calibração precisa ser refeita.
    """
    g = carregar_geometria()
    medido = json.loads(Path(caminho).read_text(encoding="utf-8"))
    g["objp_tabuleiro"] = np.array(medido["cantos_tabuleiro_mm"], np.float32)
    if medido["cantos_marcador_mm"] is not None:
        g["objp_marcador"] = np.array(medido["cantos_marcador_mm"], np.float32)
    return g


def _ordem_invertida(gray, c, nx, ny):
    """True se a ordem começa no lado oposto: as casas de (i + j) par devem ser pretas."""
    grade = c.reshape(ny, nx, 2)
    centros = (grade[:-1, :-1] + grade[:-1, 1:] + grade[1:, :-1] + grade[1:, 1:]) / 4
    h, w = gray.shape[:2]
    soma = [[], []]
    for j in range(ny - 1):
        for i in range(nx - 1):
            x, y = np.rint(centros[j, i]).astype(int)
            viz = gray[max(0, y - 1):min(h, y + 2), max(0, x - 1):min(w, x + 2)]
            soma[(i + j) % 2].append(float(viz.mean()))
    return np.mean(soma[0]) > np.mean(soma[1])


def cantos_tabuleiro(gray, geo, refinar=True):
    nx, ny = geo["padrao"]
    ok, c = cv2.findChessboardCornersSB(gray, (nx, ny), cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_ACCURACY)
    if not ok:
        return None
    c = c.reshape(-1, 1, 2).astype(np.float32)
    if _ordem_invertida(gray, c, nx, ny):
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
