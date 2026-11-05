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


def _reta_da_borda(gray, a, b, normal, meia_largura):
    """Reta (normal, ponto) da transição preto-branco entre os cantos a e b."""
    u = np.arange(-meia_largura, meia_largura + 1e-6, 0.25)
    pontos = a + np.outer(np.linspace(0.12, 0.88, 40), b - a)
    amostras = pontos[:, None, :] + u[None, :, None] * normal
    perfil = cv2.remap(gray, amostras[..., 0].astype(np.float32), amostras[..., 1].astype(np.float32),
                       cv2.INTER_LINEAR)
    escuro, claro = perfil[:, :4].mean(1), perfil[:, -4:].mean(1)
    meio = (escuro + claro) / 2
    borda = []
    for j in range(len(pontos)):
        if claro[j] - escuro[j] < 25:
            continue
        sobe = np.where((perfil[j, :-1] < meio[j]) & (perfil[j, 1:] >= meio[j]))[0]
        if len(sobe) == 0:
            continue
        i = sobe[np.argmin(np.abs(u[sobe]))]
        x = u[i] + (meio[j] - perfil[j, i]) / (perfil[j, i + 1] - perfil[j, i]) * 0.25
        borda.append(pontos[j] + x * normal)
    if len(borda) < 8:
        return None
    borda = np.array(borda)
    for _ in range(3):
        centro = borda.mean(0)
        direcao_normal = np.linalg.svd(borda - centro)[2][1]
        desvio = (borda - centro) @ direcao_normal
        fica = np.abs(desvio) < max(0.3, 2.5 * desvio.std())
        if fica.sum() < 6:
            break
        borda = borda[fica]
    centro = borda.mean(0)
    return np.linalg.svd(borda - centro)[2][1], centro


def _refinar_pelas_bordas(gray, cantos):
    """Cantos como interseção das retas ajustadas às quatro bordas do marcador.

    O cornerSubPix foi feito para cantos em X. No canto externo do marcador, em L, o
    resultado depende do borrão e da perspectiva. A borda inteira é mais estável.
    Devolve None se alguma borda não puder ser medida (por exemplo, coberta pela mão).
    """
    g = gray.astype(np.float32)
    q = cantos.reshape(4, 2).astype(np.float64)
    centro = q.mean(0)
    lado = np.linalg.norm(q - np.roll(q, -1, axis=0), axis=1).mean()
    meia_largura = max(2.5, min(6.0, 0.4 * lado / 6))   # dentro da borda preta, de 1 bit
    retas = []
    for k in range(4):
        a, b = q[k], q[(k + 1) % 4]
        d = (b - a) / np.linalg.norm(b - a)
        normal = np.array([-d[1], d[0]])
        if normal @ ((a + b) / 2 - centro) < 0:
            normal = -normal
        reta = _reta_da_borda(g, a, b, normal, meia_largura)
        if reta is None:
            return None
        retas.append(reta)
    saida = []
    for k in range(4):
        (n1, c1), (n2, c2) = retas[k - 1], retas[k]
        saida.append(np.linalg.solve(np.array([n1, n2]), np.array([n1 @ c1, n2 @ c2])))
    saida = np.array(saida)
    if np.abs(saida - q).max() > 3:
        return None
    return saida.reshape(-1, 1, 2).astype(np.float32)


_DETECTORES = {}


def cantos_marcador(gray, geo, refinar_bordas=True):
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
            c = c.reshape(-1, 1, 2).astype(np.float32)
            refinado = _refinar_pelas_bordas(gray, c) if refinar_bordas else None
            return c if refinado is None else refinado
    return None
