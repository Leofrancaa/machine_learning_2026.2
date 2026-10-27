# v3: resultados finais (18/10/2026)

Configuração escolhida pelas ablações ([08-ablacoes.md](08-ablacoes.md)): a v2 ([07-higiene-v2.md](07-higiene-v2.md)) com **normalização de intensidade por imagem** (B) e **score restrito a uma máscara média dos pulmões** (C1). Execução completa: 40 épocas, 128 px, latente 8, β = 1, 3 sementes e todos os baselines, para PA e para AP + PA. O teste foi olhado **uma única vez**, nesta configuração; depois disso ela não mudou.

```bash
uv run python -m module_vae.main --normalize --lung-mask --version v3 --label v3-final
uv run python -m module_vae.main --views AP,PA --normalize --lung-mask --version v3 --label v3-final
```

## Linha do tempo das versões (só PA, teste)

| | v1 | v2 | **v3** |
|---|---|---|---|
| O que mudou | pipeline do RSNA | + higiene (adultos, sem duplicata) | + normalização + máscara dos pulmões |
| Teste (opacidades) | 1.285 (203) | 1.226 (191) | 1.226 (191) |
| σ escolhido na validação | 10 | 8 | 4 |
| **AUC do VAE** | 0,757 | 0,769 | **0,804** |
| AUC do VAE, 3 sementes | 0,757 ± 0,002 | 0,765 ± 0,009 | 0,801 ± 0,033 |
| AP (acaso: 0,16) | 0,449 | 0,459 | **0,541** |
| Erro pixel a pixel ingênuo (σ = 0) | 0,689 | 0,700 | 0,787 |
| Sensibilidade / especificidade no limiar p95 | 29% / 96% | 29% / 95% | **46% / 94%** |
| F1 no limiar p95 | — | 0,37 | **0,52** |
| Acurácia no limiar p95 ("normal para todos": 0,844) | — | 0,847 | 0,866 |
| "Anormal sem opacidade" × normal | 0,694 | 0,699 | **0,739** |
| Fila: posição média de uma opacidade (de 51) | 27 | 27 | **23,5** |
| Fila: opacidades entre as 10 primeiras (de 1,0) | 3,7 | 3,5 | **4,5** |
| precision@10 / @25 / @100 | — | 0,90 / 0,84 / 0,51 | **1,00 / 0,92 / 0,69** |
| Pixel mais quente dentro da caixa | 25,1% | 13,6% | **35,6%** |
| Acaso da localização | 7,3% | 7,4% | 17,6% (dentro da máscara) |
| **Localização acima do acaso** | **17,8 p.p.** | 6,2 p.p. | **18,0 p.p.** |
| Score × área da lesão (Spearman) | −0,14 | −0,08 | −0,05 |

No limiar p95 da v3: 88 opacidades encontradas, 103 perdidas, 61 falsos alarmes entre 1.035 normais.

## Comparação de métodos (só PA, AUC no teste)

| Método | Usa rótulos? | v1 | v2 | **v3** |
|---|---|---|---|---|
| Brilho médio | não | 0,49 | 0,49 | 0,50 |
| Imagem média | não | 0,66 | 0,66 | 0,70 |
| PCA (8 componentes) | não | 0,745 | 0,748 | **0,808** |
| Autoencoder comum (3 sementes) | não | 0,757 | 0,755 | **0,805 ± 0,017** |
| **VAE** (3 sementes) | não | 0,757 | 0,765 | **0,801 ± 0,033** |
| Heurística "nível de detalhe" | sim | 0,55 | 0,56 | 0,58 |
| CNN supervisionada | sim | 0,825 | 0,875 | 0,857 |

Na v3, os baselines de reconstrução usam a mesma normalização e a mesma máscara do VAE, para a comparação ser justa.

## O atalho AP/PA (AP + PA, teste)

| | v1 | v2 | **v3** |
|---|---|---|---|
| AUC geral | 0,788 | 0,813 | 0,870 |
| AUC só entre exames AP | 0,664 | 0,718 | **0,825** |
| AUC só entre exames PA | 0,763 | 0,757 | 0,818 |
| **O score separa AP de PA entre os normais** | 0,701 | 0,693 | **0,657** |
| Pixel mais quente na caixa / acaso | 11,6% / 12,6% | 11,2% / 12,9% | **38,7% / 27,8%** |
| VAE / autoencoder comum (3 sementes) | 0,791 / 0,802 | 0,810 / 0,812 | 0,873 / 0,864 |
| CNN supervisionada | 0,936 | 0,947 | 0,955 |

Com a normalização, **o atalho enfraqueceu, mas não sumiu**: o score ainda separa AP de PA entre os normais (0,66). E, pela primeira vez, a localização no AP + PA fica **acima do acaso** (11 pontos), em vez de no nível dele.

## O que melhorou de verdade

- **A fila.** Uma opacidade sobe, em média, da posição 51 para 23,5 (v1: 27). Entre os 10 primeiros exames há 4,5 opacidades, contra 1,0 na ordem de chegada.
- **O alarme.** No mesmo limiar sem rótulos (p95 dos normais), a sensibilidade sobe de 29% para 46%, com especificidade de 94%.
- **Alterações nunca vistas.** "Anormal sem opacidade" × normal: de 0,69 para 0,74.
- **A distância para a CNN** cai de 0,07 para 0,05.
- **Os mapas.** Eles não acendem mais em ombros, cantos e marcadores de texto, o que importa para a demo.

## O que precisa ser dito com honestidade

1. **A melhoria não é do "V".** A normalização e a máscara ajudaram **todos** os métodos de reconstrução por igual: na v3, VAE (0,801), autoencoder comum (0,805) e PCA (0,808) estão empatados. O achado da v1 continua: **o VAE não detecta melhor que um autoencoder comum**. O que melhorou foi o pré-processamento. O que o "V" acrescenta é o espaço latente organizado (gerar exames, interpolar), mostrado na demo.
2. **O ganho real de localização é pequeno.** 35,6% parece muito melhor que os 25,1% da v1, mas, dentro da máscara, o acaso é 17,6%. Descontando o acaso, são **18,0 pontos** na v3, contra 17,8 na v1. A máscara **limpou** os mapas; não os tornou duas vezes mais precisos.
3. **Mais variação entre sementes.** O VAE da v3 varia ± 0,033 no teste (v2: ± 0,009). O modelo salvo (semente 0, AUC 0,804) está perto da média.
4. **O viés de equipamento continua.** O equipamento 0,168 (um dos "claros") segue com a pior AUC: 0,725, contra 0,84 no 0,143. A normalização não eliminou a diferença.
5. **O score continua sem medir gravidade** (correlação −0,05 com a área da lesão).
6. **A CNN caiu de 0,875 para 0,857.** Ela é treinada uma única vez: é provavelmente ruído.
7. **Com a máscara, o erro ingênuo quase alcança o suavizado** (0,787 contra 0,804), e o σ escolhido cai de 10 para 4. Grande parte do "ruído de alta frequência" que a suavização compensava vinha de fora dos pulmões.

## Limitações (atualizadas)

- **Máscara média, não por exame:** em pacientes posicionados de forma diferente da média, a máscara pega parte do pescoço ou corta a borda de um pulmão. A segmentação por exame (C2) ficou como trabalho futuro ([08](08-ablacoes.md)).
- **Um único hospital/fonte** (NIH): o limiar e a máscara precisariam ser recalibrados em outro lugar.
- **Só PA no resultado principal**; o atalho AP/PA foi reduzido, mas não eliminado.
- **As ablações só enxergam efeitos grandes** (3 sementes, ganho mínimo de 0,02 de AUC). Melhorias menores podem existir sem que esse desenho consiga mostrá-las.
- Não é um dispositivo médico.

## Demo

A demo (`interfaces/`) foi regenerada com os modelos da v3 (`make export`, a partir de `rsna_pa_v3` e `rsna_ap_pa_v3`). O export aplica o pré-processamento de cada modelo; a imagem "original" mostrada no site é o exame normalizado, como o modelo o vê, e os mapas ficam dentro da máscara dos pulmões.

## Arquivos

| | |
|---|---|
| Relatórios | `reports/rsna_pa_v3`, `reports/rsna_ap_pa_v3` |
| Modelos | `models/rsna_pa_v3`, `models/rsna_ap_pa_v3` (metadados e `lung_mask.npy`; os pesos ficam fora do git) |
| MLflow | `v3-final`, nos experimentos `rsna_pa` e `rsna_ap_pa`, ao lado de `v1-linha-de-base` e `v2-linha-de-base` |
