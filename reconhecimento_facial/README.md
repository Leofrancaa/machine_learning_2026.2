# reconhecimento_facial

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Controle de acesso por reconhecimento facial com OpenCV e DeepFace

## Resumo

Cada rosto é comparado com as fotos das pessoas autorizadas. Se a pessoa for
reconhecida, o acesso é **liberado (verde)**; caso contrário, é **negado (vermelho)**.

1. **Cadastro** (`main.py`): o DeepFace detecta o rosto de cada foto autorizada e gera
   um embedding de 512 números com o Facenet512. Os embeddings formam o banco de rostos.
2. **Reconhecimento** (`predict.py`): o rosto novo vira embedding e é comparado com o banco
   pela distância cosseno. O acesso é liberado se a menor distância for até 0,30.
3. **Acesso**: pela webcam (`inference.py`) ou, sem webcam, pela portaria virtual no
   navegador (`portaria.py`).

## Bibliotecas e tecnologias

| Biblioteca | Uso no projeto |
| --- | --- |
| DeepFace | Detecção (YuNet), embeddings (Facenet512) e threshold calibrado |
| TensorFlow / tf-keras | Backend que roda o Facenet512 |
| OpenCV | Leitura das fotos, webcam e desenho dos bounding boxes |
| NumPy e pandas | Embeddings e banco de rostos (`face_database.pkl`) |
| http.server (Python) | Servidor local da portaria, sem dependências extras |
| Three.js | Animação 3D da portaria no navegador |
| loguru, tqdm | Logs e barra de progresso do cadastro |
| uv, ruff, pytest | Ambiente e dependências, lint e testes |

## Como executar

### 1. Pré-requisitos

- [uv](https://docs.astral.sh/uv/) instalado (ele baixa o Python 3.12 sozinho, se precisar):

  ```bash
  pip install uv
  ```

- Conexão com a internet na primeira execução: o DeepFace baixa os pesos do Facenet512 e
  do detector YuNet (ficam em cache em `~/.deepface/`), e a página da portaria carrega o
  Three.js pelo jsDelivr.
- Webcam, só para o modo do passo 5. Sem webcam, use a portaria virtual (passo 6).

### 2. Instalar o ambiente

A partir da raiz do repositório:

```bash
cd reconhecimento_facial
uv sync
```

O `uv sync` cria a `.venv` com o Python 3.12 e instala as versões travadas no `uv.lock`.

### 3. Adicionar as fotos das pessoas autorizadas

Crie `data/raw/authorized/` (a pasta `data/` é ignorada pelo Git) com uma subpasta por
pessoa. O nome da subpasta é o nome exibido no acesso:

```
data/raw/authorized/
├── maria/
│   ├── 1.jpg
│   └── 2.jpg
└── joao/
    └── 1.png
```

Use fotos de frente, bem iluminadas e só com a pessoa. De 3 a 5 fotos por pessoa deixam
o reconhecimento mais estável. Formatos aceitos: JPG, PNG, BMP e WEBP (fotos HEIC do
iPhone precisam ser convertidas).

Se for usar só a portaria virtual, este passo e o próximo são opcionais: o cadastro
também pode ser feito pelo navegador.

### 4. Gerar o banco de rostos

```bash
uv run python -m module_reconhecimento.main
```

Gera `models/face_database.pkl` e `models/metadata.json`. Rode de novo sempre que
mudar as fotos.

### 5. Testar o acesso pela webcam

```bash
uv run python -m module_reconhecimento.inference
```

Abre a câmera e desenha o rosto em verde (liberado) ou vermelho (negado).
`Q` ou `Esc` fecha a janela.

### 6. Sem webcam: portaria virtual

```bash
uv run python -m module_reconhecimento.portaria
```

O navegador abre em `http://127.0.0.1:8000` com duas abas:

- **Cadastro**: nome e fotos da pessoa. As fotos com rosto são salvas em
  `data/raw/authorized/<nome>/` e o banco de rostos é gerado de novo.
- **Portaria**: envie a foto de quem está chegando. Uma animação em Three.js
  mostra a pessoa no totem; se o acesso for liberado o portão abre, senão aparece
  **ACESSO NEGADO**.

`Ctrl+C` no terminal encerra o servidor.

### 7. Testes e lint

```bash
uv run pytest
uv run ruff check
```

### Ajustes

Modelo, detector e threshold ficam em `module_reconhecimento/config.py`. Se uma
pessoa autorizada for negada, aumente o `THRESHOLD` e gere o banco de novo (passo 4).

## Organização do projeto

```
├── Makefile           <- Comandos de conveniência (make requirements, make lint, make test)
├── README.md          <- Este arquivo
├── pyproject.toml     <- Metadados do pacote module_reconhecimento, dependências e config do ruff
├── uv.lock            <- Versões travadas das dependências (usado pelo uv sync)
├── .python-version    <- Versão do Python usada pelo uv (3.12)
│
├── data               <- Ignorada pelo Git; criada localmente
│   └── raw
│       └── authorized <- Fotos das pessoas autorizadas, uma subpasta por pessoa
│
├── docs               <- Projeto mkdocs padrão
│
├── models             <- metadata.json e face_database.pkl (gerado no cadastro)
│
├── notebooks          <- Notebooks Jupyter
│
├── references         <- Dicionários de dados, manuais e materiais de apoio
│
├── reports            <- Análises geradas
│   └── figures        <- Gráficos e figuras gerados
│
├── tests              <- Testes com pytest
│
└── module_reconhecimento   <- Código-fonte do projeto
    │
    ├── __init__.py             <- Torna module_reconhecimento um módulo Python
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
    ├── portaria.py             <- Servidor da portaria virtual (cadastro e acesso por foto)
    │
    ├── web
    │   └── portaria.html       <- Página de cadastro e animação 3D da portaria (Three.js)
    │
    ├── modeling
    │   ├── __init__.py
    │   ├── predict.py          <- Compara os rostos com o banco e decide o acesso
    │   └── train.py            <- Gera e salva o banco de embeddings e os metadados
    │
    └── plots.py                <- Desenha os bounding boxes (verde/vermelho) no vídeo
```

--------
