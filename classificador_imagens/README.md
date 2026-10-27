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

## Como executar

### 1. Pré-requisitos

- [uv](https://docs.astral.sh/uv/) instalado (ele baixa o Python 3.12 sozinho, se precisar):

  ```bash
  pip install uv
  ```

- Conexão com a internet na primeira execução: o CIFAR-10 (~170 MB) é baixado
  automaticamente pelo `keras.datasets` e fica em cache em `~/.keras/datasets/`.

### 2. Instalar o ambiente

A partir da raiz do repositório:

```bash
cd classificador_imagens
uv sync
```

O `uv sync` cria a `.venv` com o Python 3.12 e instala as versões travadas no `uv.lock`.

### 3. Treinar e avaliar a CNN

```bash
uv run python -m module_classificador_imagens.main
```

Baixa os dados, treina, avalia no conjunto de teste e salva `models/cnn_model.keras`,
`models/metadata.json`, `reports/training_history.csv` e as figuras em `reports/figures/`.
No Windows o TensorFlow roda na CPU, então o treino pode levar bastante tempo.

Esse passo é opcional: o modelo treinado já está versionado em `models/`.

### 4. Inferência

```bash
uv run python -m module_classificador_imagens.inference
```

Carrega `models/cnn_model.keras`, classifica 9 imagens de teste sorteadas e salva a grade
com as previsões em `reports/figures/predictions.png`.

### 5. Notebook (opcional)

```bash
uv run jupyter lab
```

Abra `notebooks/classificador_imagens.ipynb` e execute todas as células
(*Run > Run All Cells*).

### 6. Testes e lint

```bash
uv run pytest
uv run ruff check
```

Para usar GPU no TensorFlow, é preciso WSL2 ou Linux.

## Organização do projeto

```
├── Makefile           <- Comandos de conveniência (make requirements, make lint, make test)
├── README.md          <- Este arquivo
├── pyproject.toml     <- Metadados do pacote module_classificador_imagens, dependências e config do ruff
├── uv.lock            <- Versões travadas das dependências (usado pelo uv sync)
├── .python-version    <- Versão do Python usada pelo uv (3.12)
│
├── docs               <- Projeto mkdocs padrão
│
├── models             <- cnn_model.keras e metadata.json
│
├── notebooks          <- classificador_imagens.ipynb (versão exploratória)
│
├── references         <- Dicionários de dados, manuais e materiais de apoio
│
├── reports            <- training_history.csv
│   └── figures        <- Exemplos, curvas de treino e previsões
│
├── tests              <- Testes com pytest
│
└── module_classificador_imagens   <- Código-fonte do projeto
    │
    ├── __init__.py             <- Torna module_classificador_imagens um módulo Python
    ├── config.py               <- Caminhos e configurações do projeto
    ├── dataset.py              <- Download e normalização do CIFAR-10
    ├── features.py             <- Preparação das imagens para a CNN
    ├── main.py                 <- Pipeline completo: dados, treino e avaliação
    ├── inference.py            <- Classifica imagens de teste com o modelo salvo
    ├── plots.py                <- Figuras de exemplos, curvas e previsões
    │
    └── modeling
        ├── __init__.py
        ├── pipeline.py         <- Arquitetura da CNN
        ├── train.py            <- Treino com callbacks
        ├── evaluate.py         <- Avaliação no conjunto de teste
        └── predict.py          <- Previsões com o modelo treinado
```

--------
