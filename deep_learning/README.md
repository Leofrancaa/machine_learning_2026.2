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

## Como executar

### 1. Pré-requisitos

- [uv](https://docs.astral.sh/uv/) instalado (ele baixa o Python 3.12 sozinho, se precisar):

  ```bash
  pip install uv
  ```

- Conexão com a internet: o `heart.csv` é baixado do Google Storage (pelo script
  `dataset` e pelo notebook) e o CIFAR-10 pelo `keras.datasets`.
- Opcional: [Graphviz](https://graphviz.org/download/) instalado e no `PATH`, usado só pela
  célula de `plot_model` do notebook de dados estruturados. Sem ele, apenas essa célula falha.

### 2. Instalar o ambiente

A partir da raiz do repositório:

```bash
cd deep_learning
uv sync
```

O `uv sync` cria a `.venv` com o Python 3.12 e instala as versões travadas no `uv.lock`.

### 3. Rodar o pipeline pelo terminal

O pacote `module_deep_learning` implementa, em scripts, a rede do notebook de dados
estruturados (classificação de doença cardíaca com Keras). Execute na ordem:

```bash
uv run python -m module_deep_learning.dataset
uv run python -m module_deep_learning.features
uv run python -m module_deep_learning.modeling.train
uv run python -m module_deep_learning.plots
uv run python -m module_deep_learning.modeling.predict
```

| Comando | O que faz | Gera |
| --- | --- | --- |
| `dataset` | Baixa o `heart.csv` e separa 80% treino / 20% validação | `data/raw/heart.csv`, `data/processed/train.csv` e `val.csv` |
| `features` | Normaliza as numéricas e aplica one-hot nas categóricas, com estatísticas só do treino | `data/processed/*_features.csv`, `models/preprocessing.json` |
| `modeling.train` | Treina a rede Dense(32) → Dropout(0,5) → Dense(1, sigmoid) por 50 épocas | `models/model.keras`, `reports/training_history.csv` |
| `plots` | Desenha as curvas de loss e acurácia | `reports/figures/training_curves.png` |
| `modeling.predict` | Calcula a probabilidade de doença para os pacientes de validação e para o paciente de exemplo | `data/processed/val_predictions.csv` |

Cada comando aceita `--help` para ver as opções (por exemplo, `--epochs` no treino).

### 4. Rodar os notebooks

```bash
uv run jupyter lab
```

No Jupyter, abra a pasta `notebooks/` e execute cada notebook do início ao fim
(*Run > Run All Cells*), nesta ordem:

1. `atividade_tensorflow.ipynb`: roda em segundos.
2. `structured_data_classification_from_scratch.ipynb`: treino curto (poucos minutos).
3. `classificador_imagens.ipynb`: treino da CNN; na CPU pode levar bastante tempo.

No VS Code, também dá para abrir os notebooks e escolher o kernel da `.venv` criada pelo `uv sync`.

### 5. Testes e lint

```bash
uv run pytest
uv run ruff check
```

No Windows, o TensorFlow roda na CPU. Para usar GPU, é preciso WSL2 ou Linux.

## Organização do projeto

```
├── Makefile           <- Comandos de conveniência (make requirements, make lint, make test)
├── README.md          <- Este arquivo
├── pyproject.toml     <- Metadados do pacote module_deep_learning, dependências e config do ruff
├── uv.lock            <- Versões travadas das dependências (usado pelo uv sync)
├── .python-version    <- Versão do Python usada pelo uv (3.12)
│
├── data               <- Ignorada pelo Git; criada pelo script dataset
│   ├── raw            <- heart.csv original
│   └── processed      <- Treino, validação, features codificadas e previsões
│
├── docs               <- Projeto mkdocs padrão
│
├── models             <- model.keras e preprocessing.json (gerados no pipeline)
│
├── notebooks          <- Notebooks dos exercícios (ponto de execução do projeto)
│   ├── atividade_tensorflow.ipynb
│   ├── structured_data_classification_from_scratch.ipynb
│   └── classificador_imagens.ipynb
│
├── references         <- Dicionários de dados, manuais e materiais de apoio
│
├── reports            <- training_history.csv
│   └── figures        <- training_curves.png
│
├── tests              <- Testes com pytest
│
└── module_deep_learning   <- Código-fonte do projeto
    │
    ├── __init__.py             <- Torna module_deep_learning um módulo Python
    │
    ├── config.py               <- Caminhos e configurações do projeto
    │
    ├── dataset.py              <- Baixa o heart.csv e separa treino e validação
    │
    ├── features.py             <- Normalização e one-hot das features
    │
    ├── modeling
    │   ├── __init__.py
    │   ├── predict.py          <- Probabilidade de doença cardíaca por paciente
    │   └── train.py            <- Cria e treina a rede neural com Keras
    │
    └── plots.py                <- Curvas de loss e acurácia do treino
```

--------
