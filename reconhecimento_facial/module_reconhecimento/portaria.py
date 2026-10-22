"""
Portaria virtual: cadastro de pessoas e teste de acesso pelo navegador.

Alternativa ao inference.py para quem não tem webcam. Um servidor local
entrega a página web/portaria.html e recebe as fotos enviadas por ela:

    - Cadastro: salva as fotos em data/raw/authorized/<nome>/ e gera o banco
      de rostos de novo, igual ao `module_reconhecimento.main`.
    - Portaria: reconhece o rosto da foto com o mesmo `predict` da webcam e
      responde se o acesso foi liberado ou negado.

Usa só a biblioteca padrão do Python (http.server), sem dependências novas.

Uso:
    uv run python -m module_reconhecimento.portaria
"""

import base64
import binascii
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import re
import shutil
import threading
import time
import webbrowser

import cv2
from deepface import DeepFace
from loguru import logger
import numpy as np

from module_reconhecimento.config import (
    AUTHORIZED_DIR,
    DETECTOR_BACKEND,
    DISTANCE_METRIC,
    MODEL_NAME,
    MODELS_DIR,
    THRESHOLD,
)
from module_reconhecimento.dataset import load_authorized_images, read_image
from module_reconhecimento.features import extract_faces
from module_reconhecimento.modeling.predict import load_face_database, predict
from module_reconhecimento.modeling.train import build_face_database

# Só aceita conexões do próprio computador.
HOST = "127.0.0.1"
PORT = 8000

HTML_PATH = Path(__file__).resolve().parent / "web" / "portaria.html"
DATABASE_PATH = MODELS_DIR / "face_database.pkl"
METADATA_PATH = MODELS_DIR / "metadata.json"

# Tamanho máximo de uma requisição (fotos em base64).
MAX_BODY_BYTES = 25 * 1024 * 1024

# Extensão usada para salvar cada tipo de imagem recebida.
MIME_TO_EXTENSION = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
}

# Nome da pessoa: letras (com acento), números, espaço, hífen e underline.
# Isso impede nomes como "../x", que sairiam da pasta authorized/.
NAME_PATTERN = re.compile(r"[\w\- ]{1,40}")

# Banco de rostos e metadados carregados em memória.
state = {"database": None, "metadata": None}


# =============================================
# Funções auxiliares
# =============================================


def clean_name(name: str) -> str:
    """
    Remove espaços extras do nome e confere se ele pode virar nome de pasta.
    """
    name = " ".join(str(name).split())

    if not NAME_PATTERN.fullmatch(name):
        raise ValueError("Nome inválido. Use só letras, números, espaço e hífen (até 40).")

    return name


def decode_data_url(data_url: str) -> tuple[bytes, str]:
    """
    Converte a imagem enviada pelo navegador ("data:image/jpeg;base64,...")
    em bytes e devolve também a extensão do arquivo.
    """
    match = re.fullmatch(r"data:(image/[\w.+-]+);base64,(.+)", str(data_url), re.DOTALL)

    if not match or match.group(1) not in MIME_TO_EXTENSION:
        raise ValueError("Imagem inválida. Envie JPG, PNG, WEBP ou BMP.")

    try:
        raw = base64.b64decode(match.group(2), validate=True)
    except binascii.Error as error:
        raise ValueError("Imagem corrompida.") from error

    return raw, MIME_TO_EXTENSION[match.group(1)]


def bytes_to_image(raw: bytes) -> np.ndarray:
    """
    Decodifica os bytes da imagem para o formato BGR do OpenCV.
    """
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Não foi possível abrir a imagem.")

    return image


def thumbnail(image_path: Path, size: int = 96) -> str:
    """
    Gera uma miniatura quadrada da foto, em base64, para a lista de cadastrados.
    """
    image = read_image(image_path)

    # Corta o centro da imagem em um quadrado e reduz o tamanho.
    height, width = image.shape[:2]
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    square = cv2.resize(image[top : top + side, left : left + side], (size, size))

    _, buffer = cv2.imencode(".jpg", square, [cv2.IMWRITE_JPEG_QUALITY, 80])

    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


# =============================================
# Cadastro
# =============================================


def rebuild_database() -> None:
    """
    Gera o banco de rostos de novo a partir de data/raw/authorized,
    do mesmo jeito que o `module_reconhecimento.main`.
    """
    try:
        images = load_authorized_images(AUTHORIZED_DIR)
    except FileNotFoundError:
        # Ninguém cadastrado: apaga o banco antigo.
        DATABASE_PATH.unlink(missing_ok=True)
        METADATA_PATH.unlink(missing_ok=True)
        state["database"], state["metadata"] = None, None
        logger.info("Nenhuma pessoa cadastrada.")
        return

    build_face_database(
        images=images,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR_BACKEND,
        distance_metric=DISTANCE_METRIC,
        threshold=THRESHOLD,
        database_path=DATABASE_PATH,
        metadata_path=METADATA_PATH,
    )

    state["database"], state["metadata"] = load_face_database(DATABASE_PATH, METADATA_PATH)


def list_people() -> list[dict]:
    """
    Lista as pessoas cadastradas, com o número de fotos e uma miniatura.
    """
    if not AUTHORIZED_DIR.exists():
        return []

    database = state["database"]
    people = []

    for folder in sorted(AUTHORIZED_DIR.iterdir()):
        if not folder.is_dir() or folder.name.startswith("."):
            continue

        photos = sorted(path for path in folder.iterdir() if not path.name.startswith("."))

        if not photos:
            continue

        # Fotos em que o cadastro encontrou um rosto.
        n_faces = 0 if database is None else int((database["name"] == folder.name).sum())

        people.append(
            {
                "nome": folder.name,
                "fotos": len(photos),
                "rostos": n_faces,
                "miniatura": thumbnail(photos[0]),
            }
        )

    return people


def register_person(name: str, images: list[str]) -> dict:
    """
    Cadastra uma pessoa: confere se cada foto tem rosto, salva as fotos
    válidas em data/raw/authorized/<nome>/ e gera o banco de novo.
    """
    name = clean_name(name)

    if not images:
        raise ValueError("Envie pelo menos uma foto.")

    valid = []
    rejected = 0

    for data_url in images:
        raw, extension = decode_data_url(data_url)
        faces = extract_faces(bytes_to_image(raw), MODEL_NAME, DETECTOR_BACKEND)

        if faces:
            valid.append((raw, extension))
        else:
            rejected += 1

    if not valid:
        raise ValueError(
            "Nenhum rosto encontrado nas fotos. Use fotos de frente e bem iluminadas."
        )

    person_dir = AUTHORIZED_DIR / name
    person_dir.mkdir(parents=True, exist_ok=True)

    # Nome do arquivo com data e hora, para não sobrescrever fotos antigas.
    stamp = time.strftime("%Y%m%d_%H%M%S")

    for index, (raw, extension) in enumerate(valid, start=1):
        (person_dir / f"{stamp}_{index}{extension}").write_bytes(raw)

    logger.info(f"{len(valid)} foto(s) salvas para {name}.")

    rebuild_database()

    return {"nome": name, "salvas": len(valid), "sem_rosto": rejected, "pessoas": list_people()}


def remove_person(name: str) -> dict:
    """
    Remove a pasta de fotos da pessoa e gera o banco de novo.
    """
    name = clean_name(name)
    person_dir = AUTHORIZED_DIR / name

    if not person_dir.is_dir():
        raise ValueError(f"{name} não está cadastrado.")

    shutil.rmtree(person_dir)
    logger.info(f"{name} removido do cadastro.")

    rebuild_database()

    return {"pessoas": list_people()}


# =============================================
# Portaria
# =============================================


def check_access(data_url: str) -> dict:
    """
    Reconhece o rosto da foto e decide se a portaria abre.

    Se houver mais de um rosto, vale o maior, que é quem está na frente da câmera.
    """
    if state["database"] is None:
        raise ValueError("Nenhuma pessoa cadastrada. Cadastre alguém antes de testar.")

    raw, _ = decode_data_url(data_url)
    image = bytes_to_image(raw)
    predictions = predict(image, state["database"], state["metadata"])

    result = {
        "threshold": state["metadata"]["threshold"],
        "largura": image.shape[1],
        "altura": image.shape[0],
    }

    if predictions.empty:
        logger.warning("Portaria: nenhum rosto na foto.")
        return {**result, "status": "sem_rosto"}

    face = predictions.loc[(predictions["w"] * predictions["h"]).idxmax()]
    status = "liberado" if face["authorized"] else "negado"

    logger.info(f"Portaria: {status} ({face['name']}, distância {face['distance']:.2f})")

    return {
        **result,
        "status": status,
        "nome": str(face["name"]),
        "distancia": float(face["distance"]),
        "rosto": {key: int(face[key]) for key in ("x", "y", "w", "h")},
    }


# =============================================
# Servidor web
# =============================================


class PortariaHandler(BaseHTTPRequestHandler):
    """
    Responde às requisições da página da portaria.
    """

    def send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))

        if length > MAX_BODY_BYTES:
            raise ValueError("Fotos grandes demais. Envie menos fotos por vez.")

        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self):
        if self.path == "/":
            body = HTML_PATH.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/pessoas":
            self.send_json({"pessoas": list_people()})
        else:
            self.send_json({"erro": "Página não encontrada."}, status=404)

    def do_POST(self):
        routes = {
            "/api/cadastro": lambda data: register_person(data.get("nome", ""), data.get("fotos")),
            "/api/remover": lambda data: remove_person(data.get("nome", "")),
            "/api/acesso": lambda data: check_access(data.get("foto", "")),
        }

        if self.path not in routes:
            self.send_json({"erro": "Rota não encontrada."}, status=404)
            return

        try:
            self.send_json(routes[self.path](self.read_json()))
        except (ValueError, json.JSONDecodeError) as error:
            # Erros esperados (nome inválido, foto sem rosto...) voltam para a página.
            self.send_json({"erro": str(error)}, status=400)

    def log_message(self, format, *args):
        # Silencia o log padrão de cada requisição; o loguru já mostra o que importa.
        pass


def main():

    logger.info("Carregando o modelo de reconhecimento facial...")

    if DATABASE_PATH.exists():
        state["database"], state["metadata"] = load_face_database(DATABASE_PATH, METADATA_PATH)
    else:
        DeepFace.build_model(model_name=MODEL_NAME)
        logger.info("Banco de rostos ainda vazio. Cadastre pessoas pela página.")

    server = HTTPServer((HOST, PORT), PortariaHandler)
    url = f"http://{HOST}:{PORT}"

    logger.success(f"Portaria aberta em {url} (Ctrl+C para encerrar)")

    # Abre o navegador logo depois que o servidor começar a responder.
    threading.Timer(1.0, webbrowser.open, args=(url,)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

    logger.success("Portaria encerrada.")


if __name__ == "__main__":
    main()
