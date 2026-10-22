# reconhecimento_facial

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Controle de acesso por reconhecimento facial com OpenCV e DeepFace

Cada rosto é comparado com as fotos das pessoas autorizadas. Se a pessoa for
reconhecida, o acesso é **liberado (verde)**; caso contrário, é **negado (vermelho)**.

## Ambiente

O projeto usa o [uv](https://docs.astral.sh/uv/) para gerenciar o Python (3.12) e as
dependências. Na pasta `reconhecimento_facial/`, execute:

```bash
uv sync
uv run pytest
uv run ruff check
```

## Como usar

1. Coloque as fotos das pessoas autorizadas em `data/raw/authorized/`, uma
   subpasta por pessoa. O nome da subpasta é o nome exibido no acesso:

    ```
    data/raw/authorized/
    ├── maria/
    │   ├── 1.jpg
    │   └── 2.jpg
    └── joao/
        └── 1.png
    ```

    Use fotos de frente, bem iluminadas e só com a pessoa. De 3 a 5 fotos por
    pessoa deixam o reconhecimento mais estável. Formatos aceitos: JPG, PNG, BMP
    e WEBP (fotos HEIC do iPhone precisam ser convertidas).

2. Gere o banco de rostos autorizados (rode de novo sempre que mudar as fotos):

    ```bash
    uv run python -m module_reconhecimento.main
    ```

3. Teste o acesso pela webcam (`Q` ou `Esc` fecha a janela):

    ```bash
    uv run python -m module_reconhecimento.inference
    ```

Modelo, detector e threshold ficam em `module_reconhecimento/config.py`. Se uma
pessoa autorizada for negada, aumente o `THRESHOLD` e gere o banco de novo.

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
│       └── authorized <- Fotos das pessoas autorizadas, uma subpasta por pessoa.
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
│                         module_reconhecimento and configuration for tools like black
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
└── module_reconhecimento   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes module_reconhecimento a Python module
    │
    ├── config.py               <- Caminhos e configuração do DeepFace (modelo, detector, threshold)
    │
    ├── dataset.py              <- Lista e lê as fotos das pessoas autorizadas
    │
    ├── features.py             <- Detecta os rostos e gera os embeddings com o DeepFace
    │
    ├── main.py                 <- Cadastro: gera o banco de rostos autorizados
    │
    ├── inference.py            <- Controle de acesso em tempo real pela webcam
    │
    ├── modeling
    │   ├── __init__.py
    │   ├── predict.py          <- Compara os rostos com o banco e decide o acesso
    │   └── train.py            <- Gera e salva o banco de embeddings e os metadados
    │
    └── plots.py                <- Desenha os bounding boxes (verde/vermelho) no vídeo
```

--------

