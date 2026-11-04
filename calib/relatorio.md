# Relatório de calibração

## Avisos

- **AVISO:** distância focal instável: desvio de bootstrap de 29.4 px em fx e 34.2 px em fy, acima de 2%.
- **AVISO:** ponto principal instável: desvio de bootstrap de 48.5 px em cx e 47.9 px em cy, acima de 20 px.
- **AVISO:** cobertura baixa: cantos em 27% das células da imagem, abaixo de 75%.

## Resumo

- Fotos: `data/raw/E0_stills`
- Resolução: 1280 x 960 px
- Modelo: radial k1, k2 (p1 = p2 = k3 = 0)
- Fotos usadas: 28
- RMS de reprojeção: 0.6733 px
- Inclinação: mínima 7.6°, mediana 23.7°, máxima 39.9°

## Intrínsecos

Desvio do bootstrap: 100 reamostragens dos quadros selecionados. É a medida de estabilidade a considerar; o desvio do OpenCV supõe um modelo correto e costuma ser menor.

| Parâmetro | Valor | Desvio OpenCV | Desvio bootstrap |
|---|---|---|---|
| fx (px) | 1078.866 | 13.219 | 29.416 |
| fy (px) | 1161.148 | 14.636 | 34.159 |
| cx (px) | 768.430 | 9.499 | 48.501 |
| cy (px) | 1143.483 | 10.463 | 47.910 |
| k1 | -0.229653 | 0.006551 | |
| k2 | 0.056123 | 0.003806 | |

K =

```
   1078.8662        0.0000      768.4302
      0.0000     1161.1478     1143.4833
      0.0000        0.0000        1.0000
```

Distorção (k1, k2, p1, p2, k3) = -0.229653, 0.056123, 0.000000, 0.000000, 0.000000

## Seleção

| Etapa | Fotos |
|---|---|
| Na pasta | 45 |
| Sem o tabuleiro inteiro | 0 |
| Tabuleiro a menos de 8 px da borda | 0 |
| Candidatos | 45 |
| Selecionados (até 40, sem quase duplicatas) | 28 |

## Cobertura

Quadros por faixa de inclinação:

| Faixa | Quadros |
|---|---|
| 0° a 15° | 6 |
| 15° a 30° | 16 |
| 30° a 45° | 6 |
| 45° a 60° | 0 |
| 60° a 90° | 0 |

Quadros com algum canto em cada célula da imagem (grade 8 x 6):

```
  0   0   0   0   0   0   0   0
  0   0   8  11   9   0   0   0
  0   2  22  28  18   1   0   0
  0   1   5   7   5   1   0   0
  0   0   0   0   0   0   0   0
  0   0   0   0   0   0   0   0
```

## Por quadro

Erro = RMS de reprojeção de cada foto. Média dos erros: 0.659 px; máximo: 1.067 px.
Inclinação = ângulo entre a normal do alvo e o eixo óptico.

| Foto | Tempo (s) | Erro (px) | Inclinação (°) | Distância (mm) | Laplaciano |
|---|---|---|---|---|---|
| still_0001.png |  | 0.640 | 12.9 | 607 | 1485.0 |
| still_0002.png |  | 0.570 | 15.2 | 657 | 1263.9 |
| still_0003.png |  | 0.687 | 7.7 | 632 | 1191.6 |
| still_0004.png |  | 0.771 | 24.1 | 490 | 508.3 |
| still_0005.png |  | 0.814 | 28.7 | 481 | 574.7 |
| still_0007.png |  | 0.593 | 20.8 | 495 | 531.9 |
| still_0008.png |  | 0.558 | 14.3 | 575 | 396.1 |
| still_0009.png |  | 1.067 | 19.7 | 529 | 191.2 |
| still_0011.png |  | 0.664 | 17.1 | 651 | 268.2 |
| still_0012.png |  | 0.417 | 12.5 | 641 | 443.6 |
| still_0013.png |  | 0.499 | 23.6 | 662 | 460.6 |
| still_0015.png |  | 0.603 | 30.3 | 678 | 501.1 |
| still_0017.png |  | 0.780 | 27.3 | 657 | 942.7 |
| still_0019.png |  | 0.855 | 36.0 | 712 | 1143.9 |
| still_0020.png |  | 0.876 | 39.9 | 712 | 1319.6 |
| still_0022.png |  | 0.553 | 18.4 | 696 | 274.6 |
| still_0023.png |  | 0.492 | 7.6 | 683 | 452.3 |
| still_0024.png |  | 0.438 | 13.9 | 692 | 655.8 |
| still_0025.png |  | 0.577 | 18.6 | 565 | 570.9 |
| still_0026.png |  | 0.666 | 23.7 | 632 | 617.6 |
| still_0028.png |  | 0.735 | 27.3 | 628 | 626.0 |
| still_0032.png |  | 0.728 | 28.6 | 628 | 637.6 |
| still_0035.png |  | 0.736 | 31.6 | 738 | 365.3 |
| still_0037.png |  | 0.611 | 29.9 | 723 | 554.7 |
| still_0039.png |  | 0.611 | 31.0 | 709 | 622.2 |
| still_0040.png |  | 0.650 | 30.1 | 699 | 662.8 |
| still_0043.png |  | 0.647 | 26.9 | 711 | 394.2 |
| still_0044.png |  | 0.613 | 22.2 | 672 | 247.6 |
