# Triagem de raio-X de tórax com VAE — prova de conceito

Um **Autoencoder Variacional (VAE)** treinado **somente com raios-X normais** consegue apontar quais exames fogem do padrão, indicar onde está a alteração e reordenar uma fila de exames por prioridade?

## Como rodar

O pipeline é um pacote Python (`module_vae`), no padrão
[Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/), que usa o
[RSNA Pneumonia Detection Challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge):
26.684 raios-X de adultos, com três classes (normal, opacidade pulmonar e anormal sem
opacidade) e caixas marcando as opacidades.

> ⚠️ 18/10/2026: o EDA mostrou que 1.276 desses exames são de menores de 18 anos e 5 têm
> idades impossíveis. A partir da v2, eles ficam fora do escopo (ver [Versões](#versões)).

**Pré-requisitos:**

- [uv](https://docs.astral.sh/uv/) instalado. O pipeline usa a GPU automaticamente quando existe (CUDA ou Apple Silicon/MPS).
- Conta no Kaggle, com as regras da competição aceitas na página do RSNA.
- Token da API do Kaggle salvo em `~/.kaggle/access_token` (*Settings → API → Create New Token*).

```bash
make requirements   # cria o .venv e instala as dependências (Python 3.12)
make data           # baixa o RSNA (~3,7 GB) e guarda as imagens em 128 px
make train          # pipeline completo, só incidência PA (resultado principal)
make train_ap_pa    # o mesmo com AP e PA juntas, para mostrar o atalho AP/PA
make predict        # inferência com o modelo salvo em 5 exames do teste
make ablation       # (opcional) experimentos com tamanho do latente e β
make test           # testes do pacote
```

Para conferir rapidamente se tudo funciona (2 épocas, 1 semente, sem baselines), use
`make quick`. As opções do pipeline aparecem em `uv run python -m module_vae.main --help`.

Para avaliar um raio-X qualquer (png/jpg, frontal) com o modelo treinado:

```bash
uv run python -m module_vae.inference --image caminho/do/exame.png
```

> ⚠️ 18/10/2026: a partir da v2, `make train` e `make train_ap_pa` aplicam a higiene dos
> dados e gravam numa pasta por versão (`rsna_pa_v2`, `rsna_pa_v3`...), escolhida pela
> variável `VERSION` (padrão `v2`). O pipeline não grava numa pasta que já existe, então
> as pastas da v1 ficam protegidas. Para a configuração final (v3):
>
> ```bash
> uv run python -m module_vae.main --normalize --lung-mask --version v3 --label v3-final
> uv run python -m module_vae.inference --run rsna_pa_v3 --image caminho/do/exame.png
> ```

Opções acrescentadas depois da v1 (todas em `uv run python -m module_vae.main --help`):

| Opção | O que faz |
|---|---|
| `--version v3` | pasta dos resultados (`models/rsna_pa_v3`, `reports/rsna_pa_v3`) |
| `--label nome` | nome da execução no MLflow |
| `--no-hygiene` | reproduz os dados da v1 (com crianças, duplicata e idades inválidas) |
| `--normalize` | normalização de intensidade por imagem (ablação B, usada na v3) |
| `--lung-mask` | score e mapa de calor só dentro da máscara dos pulmões (ablação C1, usada na v3) |
| `--crop-borders` | recorte das molduras pretas (ablação A, rejeitada) |
| `--seed 1` | semente do VAE principal, para medir a variação entre treinos |
| `--overwrite` | permite gravar por cima de uma versão que já existe |

Comandos acrescentados:

```bash
make mlflow         # interface do MLflow em http://127.0.0.1:5001 (experimentos rsna_pa e rsna_ap_pa)
make export         # gera os dados da demo (interfaces/public/data) a partir dos modelos da v3
uv run python -m module_vae.modeling.compare_runs --candidates abl-A-recorte,abl-B-normalizacao
                    # aplica a regra das ablações às execuções do MLflow
```

**Desenho do experimento:**

| | |
|---|---|
| Dados | só incidência PA; split por paciente: 70% treino, 15% validação, 15% teste |
| Resolução | 128 px |
| Treino do VAE | só os exames normais do treino |
| Validação | escolha da suavização do score (σ) e do limiar de decisão |
| Teste | opacidade × normal; anormal sem opacidade × normal; mapa de calor × caixas; fila simulada |
| Comparação | brilho médio, imagem média, PCA, autoencoder comum, CNN supervisionada |

**Por que só PA:** no RSNA, 78% das opacidades são exames AP (paciente no leito), contra 18%
dos normais. Um modelo treinado com as duas incidências pode aprender a diferença AP × PA
em vez da doença. `make train_ap_pa` roda o mesmo pipeline com as duas incidências para
medir esse atalho.

Cada execução salva os resultados na sua pasta (`rsna_pa` ou `rsna_ap_pa`):

| Arquivo | Conteúdo |
|---|---|
| `models/<execução>/vae.pt` | pesos do VAE |
| `models/<execução>/metadata.json` | σ do score, limiar e escala do mapa de calor, todos calibrados na validação |
| `reports/<execução>/metrics.json` | métricas no teste, outras anomalias, localização, incidência, fila e comparação de métodos |
| `reports/<execução>/test_predictions.csv` | score, decisão e posição na fila de cada exame de teste |
| `reports/<execução>/thresholds.csv` | sensibilidade e especificidade de cada limiar (validação × teste) |
| `reports/<execução>/method_comparison.csv` | AUC do VAE e das alternativas |
| `reports/<execução>/figures/*.png` | curvas de treino, escolha de σ, ROC, mapas de calor com as caixas, galeria de erros, curva de ganho |

A partir da v2, cada execução também salva `reports/<execução>/val_predictions.csv` (scores
da validação), `excluded.csv` (exames fora do escopo) e, com `--lung-mask`,
`models/<execução>/lung_mask.npy`. A tabela de limiares ganhou F1, acurácia e a referência
"normal para todos".

## Versões

Cada versão tem pastas próprias em `models/` e `reports/`, uma execução no MLflow e um
documento em `docs/`. As anteriores não são sobrescritas. A linha do tempo completa está em
[docs/README.md](docs/README.md).

| Versão | O que mudou | AUC VAE (teste, só PA) | Fila (de 51) | Documento |
|---|---|---|---|---|
| v1 | pipeline do RSNA, só PA, 128 px | 0,757 | 27 | [docs/04](docs/04-resultados-rsna.md) |
| v2 | + higiene: só adultos, sem duplicata, sem idades impossíveis; métricas novas | 0,769 | 27 | [docs/07](docs/07-higiene-v2.md) |
| v3 | + normalização de intensidade e máscara dos pulmões (escolhidas por ablação, só com a validação) | **0,804** | **23,5** | [docs/08](docs/08-ablacoes.md), [docs/09](docs/09-resultados-v3.md) |

Na v3, o autoencoder comum (0,805) e o PCA (0,808) ficam empatados com o VAE: o ganho veio
do pré-processamento, não da parte variacional. A CNN supervisionada fica em 0,857.

## Estrutura

```
├── Makefile               <- atalhos: make data, make train, make predict...
├── pyproject.toml         <- dependências e configuração do ruff
├── data
│   ├── raw/rsna           <- arquivos do Kaggle (não versionados)
│   └── processed          <- imagens redimensionadas (não versionadas)
├── models                 <- modelos treinados e metadados, uma pasta por execução
├── reports                <- métricas, tabelas e figuras, uma pasta por execução
├── tests                  <- testes do pacote (pytest)
└── module_vae             <- código do pipeline
    ├── config.py          <- caminhos, hiperparâmetros e dispositivo (CPU/GPU)
    ├── dataset.py         <- download do RSNA e leitura dos DICOMs
    ├── features.py        <- do resíduo de reconstrução ao score e ao mapa de calor
    ├── plots.py           <- figuras
    ├── main.py            <- pipeline completo (make train)
    ├── inference.py       <- inferência com o modelo salvo (make predict)
    └── modeling
        ├── architecture.py   <- VAE, loss e CNN supervisionada
        ├── split.py          <- split por paciente; treino só com exames normais
        ├── train.py          <- loop de treino e salvamento do modelo
        ├── calibration.py    <- escolha de σ e do limiar na validação
        ├── evaluate.py       <- métricas, outras anomalias, localização, incidência e fila
        ├── baselines.py      <- comparação com brilho, PCA, autoencoder comum e CNN
        ├── predict.py        <- reconstrução, score e posição na fila
        └── ablation.py       <- experimentos com hiperparâmetros
```

> ⚠️ 18/10/2026: módulos acrescentados depois da v1: `preprocessing.py` (recorte,
> normalização e máscara dos pulmões), `tracking.py` (MLflow), `export.py` (dados da demo) e
> `modeling/compare_runs.py` (regra das ablações). A demo fica em `interfaces/` e os
> insumos da apresentação em `docs/`.

## Dados

- [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge) (RSNA, NIH, Society of Thoracic Radiology e MD.ai). O uso segue as regras da competição no Kaggle.

Não é um dispositivo médico e não deve ser usado para diagnóstico.
