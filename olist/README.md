# olist

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Brazilian e-commerce public dataset

## Análise exploratória de atrasos

O projeto inclui uma EDA orientada por hipóteses para identificar, no momento da
aprovação do pagamento, fatores associados ao risco de entrega após o prazo prometido.
O código reutilizável está em `module_olist/eda.py` e o roteiro completo, com tabelas,
gráficos e conclusões, está em `notebooks/02_eda_order_delay.ipynb`.
`notebooks/olist_orders.ipynb` traz o profiling da tabela de pedidos com o
`fg-data-profiling` e gera três relatórios em `reports/`: o perfil bruto
(`olist_orders_profiling.html`), o perfil com features de tempo
(`olist_orders_profiling_enriquecido.html`) e o comparativo entre pedidos no prazo e
atrasados (`olist_orders_profiling_comparativo.html`). `notebooks/01_data_quality.ipynb`
gera `reports/profile_olist_orders_profile.html` e valida a tabela com `pandera`.

## Seleção de modelo e active learning

O treino compara XGBoost, LightGBM e Gradient Boosting com validação cruzada
estratificada, escolhe o threshold de probabilidade pelo F1 out-of-fold e avalia o
modelo escolhido num conjunto de teste separado. São salvos `models/model.pkl`,
`models/best_model.joblib` com `models/metadata.json` (para a inferência com o modelo
separado) e `models/threshold.json` com o modelo escolhido e o threshold de decisão.

A simulação de active learning roda sobre a partição de treino: começa com 10 pedidos
rotulados, consulta 5 pedidos incertos por rodada e usa no máximo 1.000 pedidos de
treino para manter o label spreading leve. O histórico de cada rodada vai para
`reports/active_learning.csv` e os exemplos escolhidos são plotados em
`reports/figures/active_learning.png`.

## Explicabilidade com SHAP

O `module_olist.explain` lê `data/interim/dataset.csv` e `models/model.pkl` e salva em
`reports/figures/` o gráfico de importância global, o beeswarm, um waterfall individual
e o summary plot clássico (`shap_summary_plot.png`). Os valores SHAP descrevem a saída
bruta do modelo, que pode estar em log-odds e não em probabilidade.

## Como executar

### 1. Pré-requisitos

- [uv](https://docs.astral.sh/uv/) instalado (ele baixa o Python 3.12 sozinho, se precisar):

  ```bash
  pip install uv
  ```

- Dataset público da Olist, baixado do Kaggle:
  [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).

### 2. Colocar os dados em `data/raw/`

Crie a pasta `data/raw/` dentro de `olist/` (ela é ignorada pelo Git) e copie os CSVs:

| Arquivo | Usado por |
| --- | --- |
| `olist_orders_dataset.csv` | Pipeline, EDA e todos os notebooks |
| `olist_order_items_dataset.csv` | Pipeline, EDA e `01_fg_data_profiling.ipynb` |
| `olist_customers_dataset.csv` | Pipeline, EDA e `01_fg_data_profiling.ipynb` |
| `olist_order_payments_dataset.csv` | Só `01_fg_data_profiling.ipynb` |
| `olist_order_reviews_dataset.csv` | Só `01_fg_data_profiling.ipynb` |

### 3. Instalar o ambiente

A partir da raiz do repositório:

```bash
cd olist
uv sync
```

O `uv sync` cria a `.venv` com o Python 3.12 e instala as versões travadas no `uv.lock`.

### 4. Rodar o pipeline (dados + features + treino)

```bash
uv run python -m module_olist.main
```

Lê os CSVs de `data/raw/`, monta o dataset por pedido com as features, salva
`data/interim/dataset.csv` e treina e avalia os modelos, gerando os arquivos em `models/`.
Os próximos passos dependem dele.

### 5. Inferência

```bash
uv run python -m module_olist.inference
uv run python -m module_olist.modeling.inference
```

O primeiro mostra a prévia de pedidos atrasados do conjunto de teste com o `model.pkl`;
o segundo usa o `best_model.joblib` e o `metadata.json`.

### 6. Explicabilidade (SHAP)

```bash
uv run python -m module_olist.explain
```

### 7. Active learning

```bash
uv run python -m module_olist.modeling.active_learning
```

### 8. Notebooks

Para abrir e rodar no Jupyter:

```bash
uv run jupyter lab
```

Ordem sugerida em `notebooks/`: `01_fg_data_profiling.ipynb`, `01_data_quality.ipynb`,
`olist_orders.ipynb`, `02_EDA.ipynb` e `02_eda_order_delay.ipynb`.

Para executar sem abrir o Jupyter (o resultado fica salvo no próprio notebook):

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_eda_order_delay.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/olist_orders.ipynb
```

### 9. Testes e lint

```bash
uv run pytest
uv run ruff check
```

## Organização do projeto

```
├── Makefile           <- Comandos de conveniência (make requirements, make data, make lint, make test)
├── README.md          <- Este arquivo
├── pyproject.toml     <- Metadados do pacote module_olist, dependências e config do ruff
├── uv.lock            <- Versões travadas das dependências (usado pelo uv sync)
├── .python-version    <- Versão do Python usada pelo uv (3.12)
│
├── data               <- Ignorada pelo Git; criada localmente
│   ├── raw            <- CSVs originais da Olist
│   └── interim        <- dataset.csv gerado pelo pipeline
│
├── docs               <- Projeto mkdocs padrão
│
├── models             <- model.pkl, best_model.joblib, metadata.json e threshold.json
│
├── notebooks          <- Profiling, qualidade dos dados e EDA
│
├── references         <- Dicionários de dados, manuais e materiais de apoio
│
├── reports            <- Relatórios HTML de profiling e active_learning.csv
│   └── figures        <- Gráficos do SHAP e do active learning
│
├── tests              <- Testes com pytest
│
└── module_olist       <- Código-fonte do projeto
    │
    ├── __init__.py             <- Torna module_olist um módulo Python
    ├── config.py               <- Caminhos e configurações do projeto
    ├── dataset.py              <- Leitura dos CSVs e montagem do dataset por pedido
    ├── features.py             <- Criação das features
    ├── eda.py                  <- Funções da análise exploratória
    ├── main.py                 <- Pipeline completo: dados, features e treino
    ├── inference.py            <- Prévia das previsões no conjunto de teste
    ├── explain.py              <- Gráficos de explicabilidade com SHAP
    ├── plots.py                <- Código para gerar visualizações
    │
    └── modeling
        ├── __init__.py
        ├── pipeline.py         <- Pipelines dos modelos de boosting
        ├── split.py            <- Divisão treino/teste
        ├── cross_validation.py <- Validação cruzada estratificada
        ├── train.py            <- Treino, seleção do modelo e do threshold
        ├── evaluate.py         <- Métricas de avaliação
        ├── predict.py          <- Previsões com o modelo treinado
        ├── inference.py        <- Inferência com best_model.joblib
        ├── interpret.py        <- Cálculo dos valores SHAP
        └── active_learning.py  <- Simulação de active learning
```

--------
