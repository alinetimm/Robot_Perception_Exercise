# Deck pose uncertainty

Projeto final de Robot Perception (FURG, 2026/2): pose de um marcador ArUco e a covariância dessa pose, validadas contra uma referência de tabuleiro impressa na mesma folha.

## Por onde começar

1. Leia `PLANO.md` (escopo, protocolo, cronograma, estrutura do paper e do vídeo).
2. Imprima `alvo/alvo_A4.pdf` em **100%**, confira a régua de 100 mm e cole num papelão rígido.
3. Abra esta pasta no Claude Code (`claude` no terminal, dentro dela) e siga `PROMPTS.md` a partir do M0. O Claude Code lê o `CLAUDE.md` automaticamente.

## Já pronto

- `alvo/alvo_A4.pdf`: tabuleiro 9 × 6 quadrados de 20 mm (8 × 5 cantos internos) e ArUco `DICT_4X4_50` id 0 de 80 mm, com régua de 100 mm.
- `alvo/geometria_alvo.json`: posição dos cantos do ArUco no referencial do tabuleiro.
- `alvo/referencia.py`: detecção do tabuleiro com ordem de cantos garantida e do ArUco. Testado em 60 poses sintéticas: a pose do tabuleiro tem erro mediano de 0,17 mm, e a do marcador concorda com a do tabuleiro (diferença mediana de 0,24°).
- `alvo/gerar_alvo.py`: regenera o alvo se você quiser mudar tamanhos.
