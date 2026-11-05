# Plano de execução: pose e incerteza visual de uma plataforma de pouso

Projeto final de Robot Perception (2026/2). Entrega em **02/12**: paper de até 6 páginas (IEEE, duas colunas) e vídeo de até 5 minutos mostrando o algoritmo funcionando.

## 1. Escopo final (o que mudou em relação à proposta)

A pergunta continua a mesma: **a covariância que um pipeline de marcador fiducial atribui à pose corresponde ao erro real dela?** Isso é o que a camada de Active Inference da tese precisa receber como precisão da observação.

| Item da proposta | Situação | Motivo |
|---|---|---|
| Calibração da câmera | **Mantido** (refazer no modo de vídeo) | A K das fotos do iPhone não vale para vídeo |
| ArUco + PnP | **Mantido**, núcleo | |
| Covariância por quadro (σ²(JᵀJ)⁻¹) | **Mantido**, núcleo | É a contribuição |
| Referência pelo tabuleiro na mesma folha | **Mantido**, núcleo | Ground truth sem simulador |
| Avaliação com NEES | **Mantido**, núcleo | |
| Rastreamento por optical flow na oclusão | **Mantido** como extensão | Aula de 04/11 |
| Variante sem marcador (features + homografia) | Opcional | Só se sobrar tempo na semana 3 |
| Simulação no Gazebo | Fora | Não cabe em 4 semanas; vira trabalho futuro |
| Vídeos de campo do Hydrone | Fora | Sem ground truth; trabalho futuro |
| Experimento de brilho (E4) | Fora | Gravação complexa: o plástico impediu a detecção do tabuleiro; trabalho futuro |

O deslocamento de escopo deve aparecer no paper como limitação: o experimento é em bancada, com a câmera fixa e o alvo movido à mão. A geometria relativa (câmera vendo uma plataforma que inclina) é a mesma do pouso, mas não há água, vento nem reflexo especular real.

## 2. Hipóteses testáveis

- **H1.** Com o ruído de canto σ medido em cena estática, a covariância analítica é consistente em inclinações moderadas: o NEES médio fica dentro do intervalo de 95% de χ²₆.
- **H2.** A consistência se degrada em inclinações fortes (> 60°) e perto da pose frontal, onde a ambiguidade do PnP planar (duas soluções do IPPE) gera erros que a covariância local não prevê.
- **H3.** Durante uma oclusão, o rastreamento dos cantos por Lucas-Kanade mantém uma estimativa utilizável por alguns quadros, e a covariância inflada com o tempo sem detecção permanece consistente.

Os resultados podem confirmar ou refutar cada uma. Refutar H1 também é resultado, desde que o paper mostre onde e por quê.

## 3. Montagem experimental

**Câmera.** A recomendação é a **webcam do notebook ou uma USB, fixa num suporte**. Ela não tem estabilização eletrônica, o que preserva a K, e permite fazer a demo ao vivo para o vídeo. A alternativa é o iPhone com o app **Blackmagic Camera** (gratuito), que permite desligar a estabilização e travar foco e exposição. O app Câmera padrão do iPhone estabiliza o vídeo, e com isso a K muda de um quadro para outro, o que inviabiliza o PnP.

**Calibração.** Refaça a calibração **na mesma resolução e no mesmo modo** em que vai gravar. Use o `calibrate.py` do exercício e as lições dele: o alvo colado num papelão rígido, e o tabuleiro inclinado de 30° a 60° em direções variadas, chegando aos cantos da imagem.

**Alvo.** Imprima `alvo/alvo_A4.pdf` **em 100% (tamanho real)**. Confira a régua de 100 mm com uma régua de verdade; se a impressão tiver escalado, ajuste `quadrado_mm` e `lado_mm` em `alvo/geometria_alvo.json`. Cole a folha num papelão grosso ou numa placa de espuma, sem bolhas nem ondulações. O tabuleiro (8 × 5 cantos internos, 20 mm) é a referência; o ArUco (80 mm, `DICT_4X4_50`, id 0) faz o papel do convés.

**Iluminação.** Luz difusa e constante.

## 4. Protocolo de gravação

Grave em 30 fps, na resolução calibrada. Nomeie os arquivos como `E<n>_<descricao>.mp4` e guarde em `data/raw/`, fora do git.

| ID | O que gravar | Duração | Para quê |
|---|---|---|---|
| E0 | Calibração: 20 a 30 quadros do tabuleiro em poses variadas (fotos ou vídeo) | | K e distorção |
| E1 | Alvo **parado** em 5 poses diferentes (distâncias de 0,5 a 1,5 m, inclinações de 0° a 60°), 10 s cada | ~1 min | σ dos cantos (ruído de detecção) |
| E2 | Movimento lento: inclinar a placa de ±30° em roll e pitch, como um convés balançando | 2 × 60 s | Avaliação principal |
| E3 | Inclinações fortes (60° a 75°) e passagens pela pose frontal | 60 s | H2 |
| E5 | Oclusão real: cobrir o ArUco com a mão por 1 a 2 s, várias vezes, deixando o tabuleiro visível | 60 s | Demo do rastreamento |

O tabuleiro precisa estar visível em todos os quadros avaliados, porque é dele que vem a referência.

## 5. Método

**Por quadro:**
1. Detectar o tabuleiro e o ArUco (`alvo/referencia.py` já garante a ordem dos cantos).
2. Pose de referência `T_ref` por PnP com os 40 cantos do tabuleiro.
3. Pose do marcador `T_m` por PnP com os 4 cantos (`SOLVEPNP_IPPE_SQUARE`, guardando também a segunda solução e a razão entre os erros das duas).
4. Pose esperada do marcador: `T_ref` composta com o deslocamento conhecido do `geometria_alvo.json`.
5. Erro: rotação geodésica `‖log(R_esp⁻¹ R_m)‖` (graus) e translação (mm), também separados por eixo.

**Covariância:**
- Jacobiana `J` (8 × 6) da reprojeção dos 4 cantos em relação a uma perturbação **local** da pose (`R ← exp([δθ]×) R`, `t ← t + δt`), calculada por diferenças finitas. Isso evita a inconsistência entre a parametrização `rvec` e o erro geodésico.
- `Σ = σ² (JᵀJ)⁻¹`, com dois modelos de σ:
  - **(a) σ fixo**, medido em E1 como o desvio-padrão dos cantos em cena estática;
  - **(b) σ residual**, estimado pelo próprio quadro (`RSS / (8 − 6)`). Com apenas 2 graus de liberdade, ele deve ser instável, e mostrar isso é parte do resultado.
- A incerteza da referência (tabuleiro, 40 cantos) é calculada da mesma forma e somada; espera-se que seja desprezível, e o paper deve mostrar isso.

**Consistência:**
- `NEES = eᵀ Σ⁻¹ e`, com `e = [δθ, δt]`. Se a covariância está correta, `NEES ~ χ²₆` (média 6).
- Relate o NEES médio (ANEES), a fração dentro do intervalo de 95% e o NEES em função da inclinação e da distância.

**Oclusão (extensão):**
- **Oclusão sintética nos vídeos E2:** pinte de cinza a região do ArUco em janelas de N quadros. O tabuleiro continua visível, então a referência existe durante a oclusão.
- Durante a oclusão, rastreie os 4 cantos com `calcOpticalFlowPyrLK` a partir do último quadro detectado, resolva o PnP nos cantos rastreados e infle Σ com o tempo sem detecção (`Σ_k = Σ_LK + k·Q`, com Q ajustado em parte dos dados e avaliado no resto).
- Métricas: erro e NEES em função do número de quadros desde a perda.

## 6. Resultados que o paper precisa ter

**Figuras:**
1. Montagem: alvo, câmera e um quadro com os eixos projetados.
2. Pipeline (diagrama).
3. Erro de rotação e translação ao longo do tempo (E2), com a banda de ±2σ prevista.
4. NEES por quadro com as linhas de χ²₆ a 95%; ANEES por condição (E2, E3) e por modelo de σ.
5. NEES e erro em função da inclinação do alvo (H2).
6. Oclusão: erro e NEES em função dos quadros desde a perda (H3).

**Tabela:** por condição, número de quadros, taxa de detecção, erro mediano (graus e mm), ANEES e % dentro do intervalo de 95%, para σ fixo e σ residual.

## 7. Estrutura do paper (6 páginas)

| Seção | Páginas | Conteúdo |
|---|---|---|
| Abstract | | 150 a 200 palavras: contexto, lacuna, método, resultado, implicação |
| I. Introduction | 0,6 | Pouso UAV-USV, por que a precisão da observação importa para a camada de decisão; contribuições |
| II. Related Work | 0,6 | Pouso visual em plataformas móveis (Polvara 2018, Baca 2019, Morales-Rodríguez 2023); variância de marcadores (Adámek 2023); PnP planar e ambiguidade (IPPE) |
| III. Method | 1,4 | Modelo de câmera e calibração; PnP do marcador e da referência; Jacobiana local e Σ; NEES; rastreamento LK |
| IV. Experimental Setup | 0,7 | Câmera, calibração (RMS), alvo, protocolo E0 a E5, sem E4 |
| V. Results | 1,4 | Figuras 3 a 6 e a tabela |
| VI. Discussion | 0,6 | Onde a covariância falha e por quê; limites da bancada; o que isso implica para a tese |
| VII. Conclusion | 0,2 | |
| References | 0,5 | |

Regras de escrita: as de `CLAUDE.md` (seção "Escrita").

## 8. Vídeo (5 minutos)

| Tempo | Conteúdo |
|---|---|
| 0:00 a 0:30 | O problema: pouso num barco, por que a confiança na pose importa |
| 0:30 a 1:00 | Montagem: câmera, alvo, o que é a referência |
| 1:00 a 2:30 | **Demo ao vivo:** eixos sobre o marcador, elipse de incerteza 3σ, valores de erro e NEES na tela, inclinando a placa até a incerteza crescer |
| 2:30 a 3:15 | **Oclusão:** cobrir o marcador com a mão, o estado passa para "RASTREADO" e a elipse cresce; ao descobrir, volta para "DETECTADO" |
| 3:15 a 4:30 | Resultados: as figuras 3 a 6, uma frase por figura |
| 4:30 a 5:00 | Limitações e próximos passos (Gazebo, campo, ligação com a tese) |

## 9. Cronograma

| Semana | Datas | Entregas |
|---|---|---|
| 1 | 04/11 a 10/11 | Alvo impresso e montado; câmera escolhida e calibrada; gravações E1, E2, E3 e E5; pipeline por quadro (M0 a M3) |
| 2 | 11/11 a 17/11 | Covariância e NEES validados no sintético; σ de E1; avaliação de E2 e E3; primeiras figuras (M4, M5) |
| 3 | 18/11 a 24/11 | Oclusão sintética + rastreamento LK; demo ao vivo gravada (M6, M7) |
| 4 | 25/11 a 02/12 | Figuras finais, paper, vídeo; folga para refazer gravações (M8, M9) |

## 10. Riscos

| Risco | Sinal | O que fazer |
|---|---|---|
| Calibração ruim | f instável entre subconjuntos; RMS > 1 px | Mais poses inclinadas, alvo rígido; comparar com o EXIF ou com o datasheet |
| Webcam com autofoco ou autoexposição variando | Bordas borradas, NEES oscilando com a luz | Travar foco e exposição com `v4l2-ctl` (Linux) |
| Tabuleiro sai do quadro em inclinações fortes | Queda da taxa de detecção em E3 | Afastar a câmera; usar o ArUco maior se precisar |
| Ambiguidade do IPPE | Saltos de ~180° na rotação do marcador | Registrar as duas soluções; é material para H2, não um bug a esconder |
| NEES muito acima de 6 em tudo | σ subestimado | Medir σ em E1 por distância; incluir o erro de calibração |
| Atraso na semana 3 | | A extensão LK é a primeira a sair; o núcleo (H1, H2) já sustenta o paper |
