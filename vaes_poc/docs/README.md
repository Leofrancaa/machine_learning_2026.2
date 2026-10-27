# Insumos para a apresentação

Anotações das discussões feitas durante o desenvolvimento da PoC, para usar no preparo da apresentação.

| Arquivo | Conteúdo | Itens do barema |
|---|---|---|
| [01-problema.md](01-problema.md) | O problema, por que um VAE, o que a PoC não resolve, frase de abertura | 1 |
| [02-natureza-dos-vaes.md](02-natureza-dos-vaes.md) | Como um VAE se compara a classificação, regressão, CNN e reconhecimento facial; casos de uso | 1, 2 |
| [03-metricas.md](03-metricas.md) | Loss × métricas da tarefa, precision/recall, armadilhas | 3, 4 |
| [04-resultados-rsna.md](04-resultados-rsna.md) | Dataset, desenho do experimento, resultados, atalho AP/PA | 4 |
| [05-decisoes.md](05-decisoes.md) | Por que o RSNA, por que a "opção A", Docker, plano da demo | 3, 5 |
| [06-eda-rsna.md](06-eda-rsna.md) | EDA do dataset: qualidade dos dados, 7 perguntas com hipóteses, o que precisa de tratamento | 3, 4 |
| [07-guia-do-projeto.md](07-guia-do-projeto.md) | Guia completo: finalidade, VAE, dados, score, resultados, interface e próximos passos | 1–5 |
| [07-higiene-v2.md](07-higiene-v2.md) | v2: higiene dos dados (adultos, duplicata, idades), métricas novas, variação entre sementes | 3, 4 |
| [08-ablacoes.md](08-ablacoes.md) | Ablações A, B e C1 com regra fixada antes de rodar; o que entra na v3 | 3, 4 |
| [09-resultados-v3.md](09-resultados-v3.md) | v3: resultados finais, v1 × v2 × v3, atalho AP/PA, limitações atualizadas | 4, 5 |

Barema: (1) fundamentação, (2) arquitetura/diagrama, (3) implementação, (4) avaliação experimental com métricas, erros e limitações, (5) demonstração ao vivo.

## Linha do tempo

Uma linha por entrega, na ordem em que aconteceram. Documentos antigos não são reescritos: quando algo fica desatualizado, ganha uma nota datada logo abaixo, apontando para o documento novo.

| Data | Versão | O que mudou | Documento |
|---|---|---|---|
| 18/10/2026 | v1 | Pipeline com o RSNA (128 px, só PA); atalho AP/PA medido; notas da apresentação | [01](01-problema.md) a [05](05-decisoes.md) |
| 18/10/2026 | EDA | Qualidade dos dados, 7 perguntas, tratamentos propostos | [06](06-eda-rsna.md) |
| 18/10/2026 | v2 | Higiene (adultos, sem duplicata, idades inválidas), F1, precision@k, AUC por equipamento, localização na validação; variação entre sementes medida | [07](07-higiene-v2.md) |
| 18/10/2026 | ablações | Regra fixada antes; A (recorte) rejeitada, B (normalização) e C1 (máscara) aceitas, B + C1 escolhida; C2 fora do escopo | [08](08-ablacoes.md) |
| 18/10/2026 | v3 | Normalização + máscara dos pulmões; AUC PA 0,804, fila 51 → 23,5; o VAE segue empatado com o autoencoder comum e o PCA; demo regenerada | [09](09-resultados-v3.md) |
