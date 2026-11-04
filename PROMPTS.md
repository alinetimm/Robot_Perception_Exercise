# Prompts para o Claude Code, por etapa

Use um prompt por sessão, nesta ordem. Antes de cada um, rode `git status` e faça commit do que estiver pronto. Para as etapas M3 em diante, comece no **modo de planejamento** (Shift+Tab duas vezes) e só aprove o plano depois de lê-lo. Cada etapa tem um critério de pronto: não passe para a próxima sem ele.

---

## M0. Ambiente (semana 1)

```
Leia CLAUDE.md e PLANO.md. Crie um ambiente virtual em .venv, um requirements.txt com as dependências
do CLAUDE.md, inicialize o git com um .gitignore adequado (data/, .venv/, __pycache__/) e rode um teste
rápido que importa alvo/referencia.py e detecta o tabuleiro e o marcador em alvo/alvo_A4_preview.png.
Me diga a versão do OpenCV instalada.
```
**Pronto quando:** o teste detecta os 40 cantos e o marcador id 0.

## M1. Gerador sintético e testes (semana 1)

```
Crie src/synthetic.py: dado K, distorção, pose (R, t) e a imagem do alvo (alvo/alvo_A4_preview.png,
escala em geometria_alvo.json), renderiza o quadro por homografia e opcionalmente desloca os cantos
detectados com ruído gaussiano de sigma conhecido. Crie tests/test_pnp.py que verifica, em 50 poses
aleatórias (inclinação até 60 graus, distância 400 a 1500 mm), que o PnP do tabuleiro recupera a pose
com erro < 0.5 graus e < 2 mm, e que a pose do marcador esperada a partir do tabuleiro coincide com a
do próprio marcador. Rode os testes.
```
**Pronto quando:** os testes passam.

## M2. Calibração da câmera de vídeo (semana 1)

```
Vou usar a câmera <webcam / iPhone com Blackmagic Camera> em <resolução>. Adapte o calibrate.py do
exercício de calibração (vou colocá-lo em calib/) para ler quadros de um vídeo em data/raw/E0_*.mp4,
selecionar até 40 quadros nítidos e com poses diversas (descartar borrados pela variância do Laplaciano
e quase duplicados), calibrar com o modelo radial k1,k2 e salvar calib/camera.yaml e
calib/relatorio.md com RMS, K, distorção, erro por quadro e a cobertura de poses (inclinação por
quadro). Avise se a inclinação máxima for menor que 30 graus.
```
**Pronto quando:** RMS < 1 px, fx ≈ fy, ponto principal perto do centro e inclinações acima de 30°.

## M3. Pipeline por quadro (semana 1 e 2)

```
Implemente scripts/02_process_video.py: para cada quadro de um vídeo, detecta tabuleiro e marcador
com alvo/referencia.py, calcula a pose de referência (40 cantos) e a do marcador (IPPE_SQUARE,
guardando as duas soluções e a razão entre os erros de reprojeção), a pose esperada do marcador a
partir da referência e os erros de rotação (geodésico) e translação, total e por eixo. Salve um
Parquet por vídeo em data/proc/ com uma linha por quadro (incluindo quadros sem detecção) e,
opcionalmente, um vídeo com os eixos desenhados. Teste antes no sintético (M1).
```
**Pronto quando:** roda em E2 e o erro mediano é pequeno e plausível; o vídeo de overlay mostra os eixos no lugar certo.

## M4. Covariância e NEES (semana 2)

```
Implemente src/covariance.py com a Jacobiana local por diferenças finitas (ver CLAUDE.md), a
covariância para sigma fixo e para sigma residual, e o NEES com o erro na mesma parametrização.
Inclua a covariância da referência e verifique se é desprezível. Valide no sintético: com ruído
gaussiano de sigma conhecido nos cantos, o ANEES em 2000 quadros deve ficar dentro do intervalo de 95%
de chi2(6)/N. Mostre o histograma do NEES contra a densidade chi2(6).
```
**Pronto quando:** o teste sintético é consistente com σ fixo. Se não for, o erro está no código, não nos dados.

## M5. Avaliação nos vídeos (semana 2)

```
Use E1 (estático) para estimar sigma dos cantos do marcador por pose e no total (desvio-padrão dos
cantos em torno da média de cada trecho parado). Depois rode a avaliação em E2, E3 e E4 com os dois
modelos de sigma e salve results/evaluation.json com, por condição: quadros, taxa de detecção, erro
mediano e p90 (graus, mm), ANEES e fração dentro do intervalo de 95%. Gere as figuras 3, 4 e 5 do
PLANO.md em figures/.
```
**Pronto quando:** `results/evaluation.json` existe e as figuras saem de script.

## M6. Oclusão e rastreamento (semana 3)

```
Implemente a oclusão sintética descrita no PLANO.md (pintar a região do ArUco em janelas de N quadros
em E2, com o tabuleiro visível) e o rastreamento dos 4 cantos com calcOpticalFlowPyrLK a partir do
último quadro detectado, com PnP nos cantos rastreados e Sigma inflado com o tempo sem detecção.
Ajuste Q em metade das janelas e avalie na outra metade. Salve results/occlusion.json e a figura 6.
```
**Pronto quando:** a figura 6 mostra erro e NEES em função dos quadros desde a perda, com Q ajustado fora dos dados de avaliação.

## M7. Demo ao vivo (semana 3)

```
Crie scripts/live_demo.py: abre a câmera calibrada, desenha os eixos do marcador, a elipse de 3 sigma
da posição projetada na imagem, o estado (DETECTADO / RASTREADO / PERDIDO) e, quando o tabuleiro está
visível, o erro e o NEES do quadro. Tecla G grava o vídeo em video/. Mantenha pelo menos 20 fps.
```
**Pronto quando:** a demo roda e a oclusão com a mão mostra a transição de estado e a elipse crescendo.

## M8. Paper (semana 4)

```
Crie paper/ com o template IEEEtran (conference, duas colunas), a estrutura de seções do PLANO.md e
paper/numbers.tex gerado por script a partir de results/*.json. Escreva os rascunhos de Method e
Experimental Setup a partir do código e do relatório de calibração, seguindo as regras de escrita do
CLAUDE.md. Não escreva números à mão. Deixe Introduction, Discussion e Conclusion como tópicos para eu
escrever.
```
**Pronto quando:** compila em 6 páginas ou menos e todo número vem de `numbers.tex`.

## M9. Vídeo (semana 4)

```
Com base no PLANO.md (seção 8), nas figuras e nos trechos em video/, escreva video/roteiro.md com a
fala de cada trecho (no máximo 5 minutos lidos em ritmo calmo) e a lista de cortes. Gere com ffmpeg
os trechos das figuras em 1080p para eu montar.
```
