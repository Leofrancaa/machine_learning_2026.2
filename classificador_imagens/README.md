# classificador_imagens

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Pipeline ponta a ponta de classificação de imagens do CIFAR-10 com CNN

## Resumo

Classifica as 60 mil imagens 32x32 do CIFAR-10 em 10 classes (avião, automóvel, pássaro,
gato, cervo, cachorro, sapo, cavalo, navio e caminhão) com uma CNN.

- `main.py`: carrega e normaliza os dados, treina a CNN e avalia no teste. Salva o modelo
  (`models/cnn_model.keras`), os metadados, o histórico por época e as figuras.
- `inference.py`: carrega o modelo salvo e classifica 9 imagens de teste sorteadas.
- A CNN tem data augmentation (flip e translação), blocos de convolução com batch
  normalization e dropout, e treina com early stopping e redução da learning rate.

```bash
uv run python -m module_classificador_imagens.main
uv run python -m module_classificador_imagens.inference
```

## Bibliotecas e tecnologias

| Biblioteca | Uso no projeto |
| --- | --- |
| TensorFlow / Keras | Download do CIFAR-10, CNN, callbacks de treino e modelo `.keras` |
| NumPy | Normalização dos pixels e sorteio das imagens de inferência |
| pandas | Histórico de treino em CSV e tabela de previsões |
| Matplotlib | Figuras de exemplos, curvas de treino e previsões |
| loguru | Logs do pipeline |
| Jupyter | Notebook com a versão exploratória do classificador |
| uv, ruff, pytest | Ambiente e dependências, lint e testes |

## Ambiente

O projeto usa o [uv](https://docs.astral.sh/uv/) para gerenciar o Python (3.12) e as
dependências. Na pasta `classificador_imagens/`, execute:

```bash
uv sync
uv run pytest
uv run ruff check
```

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
│                         module_classificador_imagens and configuration for tools like black
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
└── module_classificador_imagens   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes module_classificador_imagens a Python module
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

