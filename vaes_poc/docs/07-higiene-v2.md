# v2: higiene dos dados e novas métricas (18/10/2026)

Primeira etapa depois do EDA ([06-eda-rsna.md](06-eda-rsna.md)): corrigir os problemas de dados encontrados, medir mais coisas e treinar de novo, **mudando só isso**. A v1 ([04-resultados-rsna.md](04-resultados-rsna.md)) continua nas pastas dela (`reports/rsna_pa`, `reports/rsna_ap_pa`); a v2 está em `reports/rsna_pa_v2` e `reports/rsna_ap_pa_v2`.

## O que mudou

### Higiene

| Problema (EDA) | Tratamento na v2 | Exames |
|---|---|---|
| Exame duplicado (mesmos pixels, dois `patientId`) | fica só o de menor `patientId` (encontrado pelo hash dos pixels) | 1 |
| Idades impossíveis (148 a 155 anos) | tratadas como ausentes; sem idade, não dá para confirmar que é adulto | 5 |
| Menores de 18 anos num dataset "de adultos" | fora do escopo da PoC | 1.276 |

- Idade, sexo e `PixelSpacing` (o equipamento) vêm dos cabeçalhos dos DICOMs, guardados em `data/processed/rsna_headers.csv`. O cache das imagens da v1 não mudou.
- **A higiene é aplicada depois do sorteio do split**, como o filtro de incidência. O split continua idêntico ao da v1, e o teste da v2 é o teste da v1 sem os exames fora do escopo. Com `--no-hygiene`, o pipeline reproduz exatamente os conjuntos da v1.
- O exame duplicado tinha as duas cópias no treino: **não havia vazamento** na v1.

| Conjuntos | v1 só PA | v2 só PA | v1 AP + PA | v2 AP + PA |
|---|---|---|---|---|
| Treino do VAE (normais) | 5.050 | 4.818 | 6.196 | 5.876 |
| Validação (opacidades) | 1.284 (202) | 1.219 (194) | 2.229 (902) | 2.094 (847) |
| Teste (opacidades) | 1.285 (203) | 1.226 (191) | 2.230 (902) | 2.100 (837) |

### Métricas novas

- **F1 e acurácia em cada limiar**, ao lado da referência "dizer normal para todos" (`thresholds.csv`).
- **precision@k e recall@k** no topo da lista ordenada (k = 10, 25, 50, 100).
- **AUC por equipamento** (`PixelSpacing`).
- **Localização também na validação**, com as caixas da validação: dá para comparar configurações sem olhar o teste.
- Todas as métricas de validação vão para o MLflow como `val_*`, e as de teste como `test_*`.

### Versões e rastreamento

- `--version v2` grava em pastas próprias, e o pipeline **se recusa a gravar numa pasta que já existe** (as da v1 ficam protegidas).
- No MLflow, os experimentos `rsna_pa` e `rsna_ap_pa` reúnem todas as versões; cada execução tem a tag `version`.

## Resultados da v2 (só PA)

| Teste | v1 | v2 |
|---|---|---|
| AUC do VAE (3 sementes) | 0,757 ± 0,002 | 0,765 ± 0,009 |
| Autoencoder comum | 0,757 | 0,755 |
| PCA | 0,745 | 0,748 |
| CNN supervisionada | 0,825 | 0,875 |
| "Anormal sem opacidade" × normal | 0,694 | 0,699 |
| Pixel mais quente dentro da caixa (acaso ~7%) | 25,1% | 13,6% |
| Fila: posição média de uma opacidade (de 51) | 27 | 27 |
| F1 / acurácia no limiar p95 ("normal para todos": 0,844) | — | 0,37 / 0,847 |
| precision@10 / @25 / @100 (acaso: 0,16) | — | 0,90 / 0,84 / 0,51 |

AUC por equipamento (teste): de 0,71 (equipamento 0,168, um dos "claros") a 0,85. O viés de equipamento do EDA aparece aqui.

**AP + PA:** AUC geral 0,813 (v1: 0,788); só entre exames AP 0,718 (v1: 0,664); o score ainda separa AP de PA entre os normais (0,69); localização no nível do acaso (11,2% contra 12,9%). **O atalho continua.**

## O que a v2 ensinou

### 1. A higiene não piorou nem melhorou o modelo

- O modelo da v1, avaliado na validação da v2 (só adultos), já dá **0,768**; o da v2, **0,770**. O pequeno ganho de AUC vem de **avaliar sem as crianças** (exames mais difíceis), não de treinar sem elas.
- A v2 é a linha de base certa por ser **mais correta** (escopo definido, sem duplicata, sem idades impossíveis), e não por ser melhor.

### 2. A variação entre treinos é bem maior do que parecia

A queda da localização (25% → 14%) parecia um efeito da higiene. Treinando a v2 com outras duas sementes (só validação):

| Semente | AUC val | Pixel mais quente na caixa |
|---|---|---|
| 0 | 0,770 | 16,0% |
| 1 | 0,785 | 27,8% |
| 2 | 0,750 | 24,7% |
| **média ± desvio-padrão** | **0,769 ± 0,018** | **22,9% ± 6,2 p.p.** |

O modelo da v1, nos mesmos exames de validação, acerta 21% a 23%. **A queda era a semente 0, não a higiene.** Isso mudou o plano das ablações: com uma semente só, elas dariam resultados aleatórios ([08-ablacoes.md](08-ablacoes.md)).

### 3. A CNN subiu 0,05

De 0,825 para 0,875 sem as crianças. Ela é treinada uma única vez, então parte disso pode ser ruído; na v3 ela fica em 0,857.

## Arquivos

| | |
|---|---|
| Relatórios | `reports/rsna_pa_v2`, `reports/rsna_ap_pa_v2`, `reports/rsna_pa_v2-s1`, `reports/rsna_pa_v2-s2` |
| Fora do escopo, por conjunto | `reports/<execução>/excluded.csv` |
| Scores da validação (para comparações pareadas) | `reports/<execução>/val_predictions.csv` |
| MLflow | `v2-linha-de-base`, `v2-semente-1`, `v2-semente-2` |
