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

- Conexão com a internet: os notebooks baixam os dados na primeira execução
  (`heart.csv` do Google Storage e o CIFAR-10 pelo `keras.datasets`).
- Opcional: [Graphviz](https://graphviz.org/download/) instalado e no `PATH`, usado só pela
  célula de `plot_model` do notebook de dados estruturados. Sem ele, apenas essa célula falha.

### 2. Instalar o ambiente

A partir da raiz do repositório:

```bash
cd deep_learning
uv sync
```

O `uv sync` cria a `.venv` com o Python 3.12 e instala as versões travadas no `uv.lock`.

### 3. Rodar os notebooks

```bash
uv run jupyter lab
```

No Jupyter, abra a pasta `notebooks/` e execute cada notebook do início ao fim
(*Run > Run All Cells*), nesta ordem:

1. `atividade_tensorflow.ipynb`: roda em segundos.
2. `structured_data_classification_from_scratch.ipynb`: treino curto (poucos minutos).
3. `classificador_imagens.ipynb`: treino da CNN; na CPU pode levar bastante tempo.

No VS Code, também dá para abrir os notebooks e escolher o kernel da `.venv` criada pelo `uv sync`.

A execução do projeto é feita pelos notebooks. Os scripts em `module_deep_learning/`
seguem a estrutura do template CCDS.

### 4. Testes e lint

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
├── docs               <- Projeto mkdocs padrão
│
├── models             <- Modelos treinados e serializados
│
├── notebooks          <- Notebooks dos exercícios (ponto de execução do projeto)
│   ├── atividade_tensorflow.ipynb
│   ├── structured_data_classification_from_scratch.ipynb
│   └── classificador_imagens.ipynb
│
├── references         <- Dicionários de dados, manuais e materiais de apoio
│
├── reports            <- Análises geradas
│   └── figures        <- Gráficos e figuras gerados
│
├── tests              <- Testes com pytest
│
└── module_deep_learning   <- Código-fonte do projeto
    │
    ├── __init__.py             <- Torna module_deep_learning um módulo Python
    │
    ├── config.py               <- Caminhos e configurações do projeto
    │
    ├── dataset.py              <- Scripts para baixar ou gerar os dados
    │
    ├── features.py             <- Código para criar as features
    │
    ├── modeling
    │   ├── __init__.py
    │   ├── predict.py          <- Inferência com os modelos treinados
    │   └── train.py            <- Treino dos modelos
    │
    └── plots.py                <- Código para gerar visualizações
```

--------
