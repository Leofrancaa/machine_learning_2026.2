from pathlib import Path

import cv2
from loguru import logger
import numpy as np
import pandas as pd

# Formatos de imagem que o OpenCV consegue abrir.
# Fotos HEIC (padrão do iPhone) precisam ser convertidas para JPG ou PNG.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_authorized_images(authorized_dir: Path) -> pd.DataFrame:
    """
    Lista as fotos das pessoas autorizadas.

    Cada pessoa tem uma subpasta com o seu nome:

        authorized/
        ├── maria/
        │   ├── 1.jpg
        │   └── 2.jpg
        └── joao/
            └── 1.png

    Uma foto solta direto em authorized/ também vale; nesse caso o nome da
    pessoa é o nome do arquivo (joao.jpg -> joao).

    Args:
        authorized_dir (Path): Pasta com as fotos das pessoas autorizadas.

    Returns:
        pd.DataFrame: Uma linha por foto, com as colunas name e image_path.
    """
    logger.info(f"Procurando fotos em: {authorized_dir}")

    rows = []

    for path in sorted(authorized_dir.rglob("*")):
        # Pula pastas e arquivos ocultos (.gitkeep, .DS_Store).
        if path.is_dir() or path.name.startswith("."):
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            logger.warning(f"Formato não suportado, arquivo ignorado: {path.name}")
            continue

        # A primeira pasta abaixo de authorized/ é o nome da pessoa.
        parts = path.relative_to(authorized_dir).parts
        name = parts[0] if len(parts) > 1 else path.stem

        rows.append({"name": name, "image_path": path})

    images = pd.DataFrame(rows, columns=["name", "image_path"])

    if images.empty:
        raise FileNotFoundError(
            f"Nenhuma foto encontrada em {authorized_dir}. "
            "Crie uma subpasta por pessoa com as fotos dela (ex.: authorized/maria/1.jpg)."
        )

    logger.info(f"Fotos encontradas: {len(images)}")
    logger.info(f"Fotos por pessoa:\n{images['name'].value_counts()}")

    return images


def read_image(image_path: Path) -> np.ndarray:
    """
    Lê uma imagem do disco em BGR, o mesmo formato dos frames da webcam no OpenCV.

    O cv2.imread falha com caminhos acentuados no Windows (ex.: authorized/joão/1.jpg).
    Por isso os bytes são lidos com o numpy e decodificados com o cv2.imdecode.

    Args:
        image_path (Path): Caminho da imagem.

    Returns:
        np.ndarray: Imagem no formato (altura, largura, 3).
    """
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Não foi possível ler a imagem: {image_path}")

    return image
