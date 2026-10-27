# Resultados no RSNA

Execução de 18/10/2026: 40 épocas, 128 px, latente 8, β = 1. Relatórios completos em `reports/rsna_pa/` e `reports/rsna_ap_pa/`.

## Dataset: RSNA Pneumonia Detection Challenge (Kaggle)

- 26.684 raios-X de adultos com rótulos, um por paciente, DICOM de 1024 px (6 a 83 anos, mediana 49).

  > ⚠️ 18/10/2026: o EDA mostrou que não são todos adultos: as idades vão de 1 a 155 anos, com 1.276 menores de 18 e 5 idades impossíveis. A partir da v2, esses exames ficam fora do escopo. Ver [06-eda-rsna.md](06-eda-rsna.md) e [07-higiene-v2.md](07-higiene-v2.md).

- Três classes: **normal** (8.851), **opacidade pulmonar** (6.012, com **9.555 caixas** marcando as lesões) e **anormal sem opacidade** (11.821).

**O atalho escondido:** a proporção de exames em incidência AP (paciente no leito, geralmente internado) muda muito entre as classes.

| Classe | % em AP |
|---|---|
| Normal | 18% |
| Anormal sem opacidade | 50% |
| Opacidade | **78%** |

Um modelo pode aprender "AP = doente" em vez da doença. Por isso o experimento principal usa **só PA**.

## Desenho do experimento

| | |
|---|---|
| Dados | só PA; split por paciente 70/15/15, estratificado por classe e incidência |
| Treino do VAE | 5.050 exames normais |
| Validação | 1.082 normais + 202 opacidades: escolha de σ e do limiar |
| Teste | 1.082 normais + 203 opacidades; + 892 "anormais sem opacidade" à parte |
| Comparação | brilho, imagem média, PCA, autoencoder comum (3 sementes), VAE (3 sementes), heurística de detalhe, CNN supervisionada |

## Resultado principal (só PA)

| Método | Usa rótulos? | AUC teste |
|---|---|---|
| Brilho médio | não | 0,49 |
| Imagem média | não | 0,66 |
| PCA (8 componentes) | não | 0,75 |
| Autoencoder comum | não | 0,757 ± 0,002 |
| **VAE** | não | **0,757 ± 0,002** |
| Heurística "nível de detalhe" | sim | 0,55 |
| CNN supervisionada | sim | 0,83 |

> ⚠️ 18/10/2026: resultados da v1. A v3 (normalização + máscara dos pulmões) chega a 0,804 no teste, com o VAE ainda empatado com o autoencoder comum e o PCA. Ver [09-resultados-v3.md](09-resultados-v3.md).

## Achados

- **O VAE empata com o autoencoder comum** e fica 0,07 abaixo da CNN supervisionada.
- **A suavização continua essencial:** erro pixel a pixel 0,69 → com σ = 10, 0,76.
- **Propriedades simples da imagem não separam as classes:** a heurística de nível de detalhe, mesmo escolhida olhando os rótulos, fica em 0,55. O sinal precisa vir da anatomia.
- **Anomalias nunca vistas:** "anormal sem opacidade" × normal dá AUC **0,69**; 19,5% deles passam do limiar (contra ~5% esperado nos normais). Suporte modesto à promessa de "detectar qualquer alteração".
- **Localização:** o ponto mais quente do mapa cai dentro da caixa do radiologista em **25%** dos casos (acaso: 7%); 13,5% do "calor" fica dentro das caixas (acaso: 7,3%). Melhor que o acaso, mas o mapa acende muito fora da lesão (bordas, ombros, abdômen).

  > ⚠️ 18/10/2026: o 25% depende muito da semente (16% a 28% entre três treinos da v2, na validação). Na v3, com a máscara dos pulmões, o mapa não acende mais fora deles: 35,6% dentro da caixa, contra 17,6% de acaso dentro da máscara. Ver [07-higiene-v2.md](07-higiene-v2.md) e [09-resultados-v3.md](09-resultados-v3.md).

- **Gravidade:** correlação score × área da lesão = **−0,14**. O score não mede gravidade.
- **Fila:** posição média de uma opacidade 51 → 27; opacidades entre os 10 primeiros 1,0 → 3,7.
- **Alarme:** no limiar sem rótulos, 29% de sensibilidade com 96% de especificidade.

## O atalho AP/PA, medido

Mesmo pipeline, com as duas incidências:

| | Só PA | AP + PA |
|---|---|---|
| AUC do VAE no teste | 0,757 | **0,788** (parece melhor) |
| AUC só entre exames AP | — | 0,66 |
| O score separa AP de PA entre os **normais**? | — | sim, AUC 0,70 |
| Ponto mais quente dentro da caixa | 25% (acaso 7%) | **12% (acaso 13%)** |
| CNN supervisionada | 0,83 | 0,94 |

O número geral sobe, mas a localização cai ao nível do acaso e o score passa a medir em parte a incidência. A CNN também ganha 0,11 com o mesmo atalho. **Número maior não é modelo melhor:** é preciso perguntar *o que* o modelo aprendeu.

## Limitações

- Mapa de calor grosseiro e que acende fora dos pulmões; restringir o score aos pulmões (segmentação) é o próximo passo natural.

  > ⚠️ 18/10/2026: feito na v3 com uma máscara média dos pulmões (ablação C1); a segmentação de cada exame (C2) ficou como trabalho futuro. Ver [08-ablacoes.md](08-ablacoes.md).

- Um único hospital/fonte (NIH); o limiar precisaria ser recalibrado em outro lugar.
- Só PA no resultado principal: o modelo não serve para exames de leito (AP) sem treino específico.
- O "V" não melhorou a detecção em relação ao autoencoder comum.
- Não é um dispositivo médico.
