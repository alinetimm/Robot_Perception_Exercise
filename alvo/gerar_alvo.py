#!/usr/bin/env python3
"""Gera o alvo de impressão (A4, escala exata) e o arquivo de geometria.

O alvo tem duas partes na mesma folha:
  - um tabuleiro de xadrez (referência): muitos cantos, pose precisa;
  - um marcador ArUco (o "convés"): 4 cantos, pose a ser avaliada.
Como os dois estão na mesma folha rígida, a pose do marcador esperada é a pose
do tabuleiro composta com um deslocamento conhecido (geometria_alvo.json).

Referencial do alvo: origem no primeiro canto interno do tabuleiro (topo à
esquerda), x para a direita, y para baixo, z entrando na folha, em mm. É o
mesmo referencial que cv2.findChessboardCorners + objp em grade produz.

Uso:
    python3 gerar_alvo.py            # gera alvo_A4.pdf, alvo_A4.png e geometria_alvo.json
Imprima em 100% (tamanho real, sem "ajustar à página") e confira a régua de
100 mm com uma régua de verdade antes de colar a folha num papelão rígido.
"""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DPI = 600
MM = DPI / 25.4                     # pixels por mm
A4 = (210.0, 297.0)                 # mm

QUADRADO = 20.0                     # lado do quadrado do tabuleiro (mm)
QUADS = (9, 6)                      # quadrados (colunas, linhas) -> 8 x 5 cantos internos (par x impar: sem ambiguidade de 180 graus)
MARCADOR = 80.0                     # lado do ArUco, incluindo a borda preta (mm)
DICIONARIO = "DICT_4X4_50"
ID_MARCADOR = 0

TAB_X0, TAB_Y0 = (A4[0] - QUADS[0] * QUADRADO) / 2, 18.0                 # canto externo do tabuleiro
MARC_X0 = (A4[0] - MARCADOR) / 2
MARC_Y0 = TAB_Y0 + QUADS[1] * QUADRADO + 28.0                            # zona branca entre os dois


def px(v):
    return int(round(v * MM))


def main():
    pasta = Path(__file__).resolve().parent
    W, H = px(A4[0]), px(A4[1])
    img = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(img)

    # tabuleiro
    for j in range(QUADS[1]):
        for i in range(QUADS[0]):
            if (i + j) % 2 == 0:
                x, y = TAB_X0 + i * QUADRADO, TAB_Y0 + j * QUADRADO
                d.rectangle([px(x), px(y), px(x + QUADRADO) - 1, px(y + QUADRADO) - 1], fill=0)

    # ArUco
    dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICIONARIO))
    lado = px(MARCADOR)
    marc = cv2.aruco.generateImageMarker(dic, ID_MARCADOR, lado, borderBits=1)
    img.paste(Image.fromarray(marc), (px(MARC_X0), px(MARC_Y0)))

    # régua de 100 mm para conferir a escala da impressão
    ry = MARC_Y0 + MARCADOR + 22.0
    rx0 = (A4[0] - 100.0) / 2
    d.line([px(rx0), px(ry), px(rx0 + 100.0), px(ry)], fill=0, width=px(0.4))
    for k in range(11):
        x = rx0 + 10.0 * k
        d.line([px(x), px(ry - (3 if k % 5 else 5)), px(x), px(ry)], fill=0, width=px(0.3))
    try:
        fonte = ImageFont.truetype("DejaVuSans.ttf", px(3.2))
    except OSError:
        fonte = ImageFont.load_default()
    d.text((px(rx0), px(ry + 2.5)),
           f"100 mm | quadrado {QUADRADO:.0f} mm | ArUco {DICIONARIO} id {ID_MARCADOR}, {MARCADOR:.0f} mm | imprimir em 100%",
           fill=0, font=fonte)

    img.save(pasta / "alvo_A4.pdf", resolution=DPI)
    img.resize((W // 4, H // 4), Image.LANCZOS).save(pasta / "alvo_A4_preview.png")

    # geometria no referencial do alvo (origem no 1º canto interno do tabuleiro)
    ox, oy = TAB_X0 + QUADRADO, TAB_Y0 + QUADRADO
    cantos_marcador = [[MARC_X0 - ox, MARC_Y0 - oy, 0.0],                      # TL (ordem do ArUco)
                       [MARC_X0 + MARCADOR - ox, MARC_Y0 - oy, 0.0],           # TR
                       [MARC_X0 + MARCADOR - ox, MARC_Y0 + MARCADOR - oy, 0.0],  # BR
                       [MARC_X0 - ox, MARC_Y0 + MARCADOR - oy, 0.0]]           # BL
    geo = {
        "unidade": "mm",
        "referencial": "origem no 1o canto interno do tabuleiro; x direita, y baixo, z para dentro da folha",
        "tabuleiro": {"cantos_internos": [QUADS[0] - 1, QUADS[1] - 1], "quadrado_mm": QUADRADO},
        "marcador": {"dicionario": DICIONARIO, "id": ID_MARCADOR, "lado_mm": MARCADOR,
                     "cantos_no_referencial_do_alvo_mm": cantos_marcador,
                     "ordem_cantos": "TL, TR, BR, BL (mesma do cv2.aruco)"},
        "pagina": {"formato": "A4", "dpi_geracao": DPI},
    }
    (pasta / "geometria_alvo.json").write_text(json.dumps(geo, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Gerados: alvo_A4.pdf, alvo_A4_preview.png, geometria_alvo.json")


if __name__ == "__main__":
    main()
