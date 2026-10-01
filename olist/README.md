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
`notebooks/profiling-orders.ipynb` e `reports/olist_orders_profiling.html` trazem
o perfil da tabela bruta de pedidos usado na `machine-learning-exs`.

Para executar a análise, coloque estes arquivos do dataset público da Olist em
`data/raw/` (a pasta é ignorada pelo Git):

- `olist_orders_dataset.csv`
- `olist_order_items_dataset.csv`
- `olist_customers_dataset.csv`

Depois, na raiz do projeto, execute:

```bash
uv run python -m module_olist.main
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_eda_order_delay.ipynb
uv run pytest
uv run ruff check
```

## SHAP explanations

After training, generate feature attribution plots with:

```bash
uv run python -m module_olist.explain
```

The command reads `data/interim/dataset.csv` and `models/model.pkl`, then saves a
global importance chart, a beeswarm chart, and an individual waterfall chart in
`reports/figures/`. SHAP values describe the model's raw output, which may use
log-odds rather than probabilities.

## Model selection and active learning

Training compares XGBoost, LightGBM, and Gradient Boosting with stratified
cross-validation, selects a probability threshold using out-of-fold F1, and
evaluates the selected model on a held-out test set. It saves both the existing
`models/model.pkl` artifact and `models/best_model.joblib` with
`models/metadata.json` for separate-model inference.

After preparing the dataset, run the active-learning simulation on the training
partition with:

```bash
uv run python -m module_olist.modeling.active_learning
```

It starts with 10 labeled orders, queries five uncertain orders per round, and
uses at most 1,000 training orders to keep label spreading manageable.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         module_olist and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── module_olist   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes module_olist a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

