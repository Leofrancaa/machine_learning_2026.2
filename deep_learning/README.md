# deep_learning

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Exercícios de deep learning com TensorFlow e Keras

## Resumo

Três notebooks que vão do básico de redes neurais até uma CNN:

| Notebook | O que faz |
| --- | --- |
| `atividade_tensorflow.ipynb` | Tensores, saída de um neurônio, ReLU, MLP 3-4-1, forward pass, threshold e loss Binary Cross-Entropy com dados de pedidos |
| `structured_data_classification_from_scratch.ipynb` | Classificação binária de doença cardíaca (303 pacientes) com features numéricas e categóricas, a partir do exemplo oficial do Keras |
| `classificador_imagens.ipynb` | CNN para as 10 classes do CIFAR-10, com data augmentation, batch normalization, dropout e early stopping (~86% de acurácia no teste) |

## Bibliotecas e tecnologias

| Biblioteca | Uso no projeto |
| --- | --- |
| TensorFlow / Keras | Tensores, camadas, treino e avaliação das redes |
| NumPy | Arrays de entrada e manipulação das imagens |
| pandas | Leitura do CSV de pacientes e conversão para `tf.data` |
| Matplotlib | Curvas de acurácia e loss, grade de imagens e previsões |
| pydot | Desenho da arquitetura com `plot_model` (exige o Graphviz instalado) |
| Jupyter | Execução dos notebooks |
| uv, ruff, pytest | Ambiente e dependências, lint e testes |

## Ambiente

O projeto usa o [uv](https://docs.astral.sh/uv/) para gerenciar o Python (3.12) e as
dependências. Na pasta `deep_learning/`, execute:

```bash
uv sync
uv run pytest
uv run ruff check
uv run jupyter lab
```

No Windows, o TensorFlow roda na CPU. Para usar GPU, é preciso WSL2 ou Linux.

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
│                         module_deep_learning and configuration for tools like black
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
└── module_deep_learning   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes module_deep_learning a Python module
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

