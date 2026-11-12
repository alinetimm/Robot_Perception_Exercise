# Incerteza da pose de um marcador fiducial

Projeto final da disciplina Robot Perception (PPGMC/FURG, 2026/2), de Aline Timm.

Um drone que pousa num barco precisa saber onde está o convés e **quanto pode confiar nessa estimativa**.
Um marcador ArUco dá a pose do convés a cada quadro, e o pipeline consegue calcular junto uma covariância.
Este projeto pergunta se essa covariância corresponde ao erro real da pose.

## A ideia

Um tabuleiro de xadrez e um marcador ArUco foram impressos **na mesma folha rígida**. O tabuleiro tem
40 cantos e dá uma pose muito mais precisa do que o marcador, que tem só 4, então ele serve de verdade de
referência sem precisar de simulador ou de sistema de captura de movimento. Em cada quadro do vídeo:

1. o tabuleiro dá a pose de referência, e dela sai a pose que o marcador deveria ter;
2. o marcador dá a sua própria pose e uma covariância de primeira ordem;
3. o erro entre as duas é comparado com a covariância pelo NEES. Se a covariância estiver certa, o NEES
   segue uma χ² com 6 graus de liberdade, de média 6.

O tabuleiro só mede o erro. A pose do marcador e a sua covariância nunca usam o tabuleiro, como seria num
pouso de verdade.

## O que o projeto faz

- **Calibração com a forma real do alvo.** A folha colada na madeira não é plana (abaula alguns
  milímetros), e tratá-la como plana enviesa a distância focal. A calibração estima, junto com a câmera, a
  posição 3D de cada canto do tabuleiro e dos cantos do marcador, e valida tudo em fotos que ficaram fora
  do ajuste.
- **Pose por quadro.** Tabuleiro por PnP; marcador por IPPE, guardando as duas soluções possíveis para
  medir a ambiguidade perto da pose frontal. Os cantos do marcador são refinados pelo ajuste de retas às
  bordas, mais estável que o sub-pixel padrão em cantos em L.
- **Quatro modelos de incerteza.** Ruído fixo entre quadros, ruído residual do próprio quadro, ruído
  efetivo e **ruído que cresce com a velocidade do marcador**. Os dois últimos são ajustados só no vídeo E1
  e testados nos outros.
- **Oclusão.** Quando a mão cobre o marcador, a pose segue por rastreamento de pontos da placa em volta
  dele (Lucas-Kanade e homografia), e a covariância cresce com o tempo sem detecção.
- **Demo ao vivo.** Mostra eixos, elipse de incerteza, erro e NEES em tempo real, e os estados
  DETECTADO, RASTREADO e PERDIDO.
- **Captura guiada.** Scripts que orientam na tela a tirada das fotos de calibração e a gravação de cada
  vídeo do protocolo.

## Resultados em uma frase cada

- O ruído medido entre quadros subestima o erro real: com ele, a covariância é otimista demais em todas
  as condições.
- A maior parte do erro que falta vem do **movimento**. Com o ruído dependente da velocidade, a mediana
  do NEES fica perto do esperado nos vídeos independentes, mas a média ainda é puxada por poucos quadros.
- Perto da pose frontal o marcador às vezes escolhe a solução errada; inclinações fortes **não** pioram
  o erro, ao contrário do que a hipótese H2 esperava.
- O rastreamento pelo entorno mantém uma pose utilizável por cerca de meio segundo de oclusão. O
  rastreamento dos 4 cantos, que era o plano original, não funciona com o marcador coberto.

Os números estão em `results/*.json` e no paper (`paper/`), sempre gerados por script.

## Estrutura

```
alvo/        alvo impresso, geometria e detecção do tabuleiro e do marcador
calib/       calibração: camera.yaml (K, distorção, validação), alvo_medido.json, relatório
src/         pipeline: pose, covariância, oclusão, estimador ao vivo
scripts/     pontos de entrada numerados (01 a 07), captura guiada e demo
tests/       testes, quase todos com dados sintéticos de pose conhecida
results/     números finais em JSON
figures/     figuras do paper (PDF vetorial e PNG)
paper/       LaTeX (IEEEtran), macros geradas e pacote para o Overleaf
data/        fotos e vídeos brutos e saídas por quadro (fora do git)
video/       gravações da demo (fora do git)
```

`PLANO.md` traz escopo, hipóteses e protocolo; `CLAUDE.md`, as regras de engenharia e de escrita.

## Como reproduzir

Ambiente (Python 3.12, OpenCV 5):

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests -q
```

Captura (webcam Logitech C270, 1280 x 960 MJPG, exposição e balanço de branco travados):

```
.venv/bin/python scripts/capture_stills.py          # fotos de calibração, com orientação na tela
.venv/bin/python scripts/record_video.py E1          # vídeos E1, E2, E3, E5, com orientação por cena
```

Processamento, na ordem:

```
.venv/bin/python calib/calibrate.py --fotos data/raw/E0_stills   # calibração e forma do alvo
.venv/bin/python scripts/01_process_frames.py                    # pose e erro por quadro
.venv/bin/python scripts/02_measure_sigma.py                     # ruído dos cantos
.venv/bin/python scripts/03_nees.py                              # covariância e NEES
.venv/bin/python scripts/04_occlusion.py                         # oclusão sintética e real
.venv/bin/python scripts/05_figures.py                           # figuras
.venv/bin/python scripts/06_numbers.py                           # macros e tabela do paper
.venv/bin/python scripts/07_overleaf.py                          # paper/overleaf.zip
```

Os scripts são determinísticos: as mesmas entradas dão as mesmas saídas. O passo 01 leva cerca de
25 minutos.

Demo ao vivo (R grava a tela em `video/`, Q sai):

```
.venv/bin/python scripts/live_demo.py
```

## Situação

Pronto: calibração, gravações, pipeline, análise de incerteza, oclusão, demo e figuras. Em andamento:
redação do paper e o vídeo de 5 minutos. A cena de brilho com filme plástico (E4) saiu do escopo, porque o
filme impedia a detecção do tabuleiro de referência.
