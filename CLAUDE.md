# CLAUDE.md

Projeto final da disciplina Robot Perception (PPGMC/FURG, 2026/2) da Aline Timm. Leia `PLANO.md` antes de qualquer tarefa: ele define escopo, hipóteses, protocolo, métricas e cronograma.

## Objetivo em uma frase

Estimar a pose de um marcador ArUco (o "convés") e a covariância dessa pose a cada quadro, e medir se a covariância corresponde ao erro real, usando como referência um tabuleiro de xadrez impresso na mesma folha rígida.

## Comunicação

- Fale com a Aline em português. Código, nomes de variáveis, legendas de figuras e o paper ficam em inglês.
- Antes de uma etapa grande, mostre o plano em poucos itens e espere a confirmação. Depois de cada etapa, diga o que foi feito, como foi verificado e o que falta.

## Estrutura

```
alvo/        alvo impresso, geometria e detecção com ordem de cantos garantida (já pronto e testado)
calib/       K e distorção da câmera usada (calib/camera.yaml), relatório da calibração
data/raw/    vídeos originais E0..E5 (fora do git)
data/proc/   saídas por quadro (CSV/Parquet) (fora do git)
src/         código do pipeline (pacote Python)
scripts/     pontos de entrada numerados: 01_..., 02_...
tests/       testes com dados sintéticos
results/     números finais em JSON (lidos pelo paper e pelas figuras)
figures/     figuras do paper (PDF vetorial + PNG)
paper/       LaTeX (IEEEtran, duas colunas)
video/       roteiro e trechos gravados
```

## Regras de engenharia

- Python 3.10, OpenCV >= 4.8 (`cv2.aruco.ArucoDetector`), NumPy, SciPy, pandas, matplotlib. Dependências em `requirements.txt`.
- Use `alvo/referencia.py` para detectar o tabuleiro e o marcador. A ordem dos cantos do tabuleiro é corrigida ali; não reimplemente.
- Unidades: mm e graus nas saídas; radianos internamente. Pose = transformação do referencial do alvo para o da câmera (`X_cam = R X_alvo + t`).
- Erro de rotação geodésico (`||log(R_a^T R_b)||`). Nunca subtraia `rvec`s.
- Covariância: Jacobiana em relação a uma perturbação local `R <- exp([dθ]x) R`, `t <- t + dt`, por diferenças finitas, ordem `[dθ, dt]`. O vetor de erro do NEES usa a mesma ordem e o mesmo lado da perturbação.
- **Teste primeiro no sintético.** Antes de rodar em vídeo real, cada componente precisa passar num teste com dados renderizados de pose conhecida (`tests/`): PnP recupera a pose; com ruído gaussiano de σ conhecido nos cantos, o ANEES fica perto de 6.
- Scripts idempotentes e determinísticos (sementes fixas). Cada script lê de `data/` ou `results/` e escreve em um único destino.
- Vídeos e dados processados não entram no git.

## Integridade dos resultados

- **Nunca invente, estime ou arredonde para "parecer bom" um número do paper.** Todo número do texto vem de `results/*.json`, gerado por script. No LaTeX, use macros geradas automaticamente (`paper/numbers.tex`) em vez de digitar valores.
- Se um resultado contradiz uma hipótese, relate como está. Refutar H1, H2 ou H3 é resultado válido.
- Referências bibliográficas: só as que a Aline forneceu ou que foram verificadas com DOI. Não crie citações.

## Figuras

- matplotlib, fonte de 8 a 9 pt, largura de 1 coluna (3,5 in) ou 2 colunas (7,16 in), PDF vetorial.
- Paleta consistente entre figuras, legível em escala de cinza; não use cor como única codificação.
- Toda figura é gerada por um script em `scripts/` a partir de `results/` ou `data/proc/`.

## Escrita (paper e textos da Aline)

- Nunca use travessão (—) nem meia-risca como pontuação.
- Sem adjetivos de valor ou superlativos ("crucial", "critical", "novel", "remarkable", "significant" fora do sentido estatístico, "robust" sem medida).
- Sem transições clichê de LLM ("Moreover", "Furthermore", "In conclusion", "It is worth noting", "delve", "seamlessly").
- Varie o comprimento das frases; uma frase curta depois de uma longa marca a ideia principal.
- Evite fórmulas de contraste paralelas ("X is A, while Y is B") e frases com dois-pontos como preparação.
- Separe observação de interpretação: "the NEES exceeds the 95% bound above 60° of tilt" (observação) vem antes de "this suggests…" (interpretação).
- O paper declara o uso de LLM na seção de agradecimentos ou métodos, conforme a política da disciplina.
