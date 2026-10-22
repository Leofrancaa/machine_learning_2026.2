import cv2
import numpy as np
import pytest

from module_reconhecimento.config import AUTHORIZED_DIR
from module_reconhecimento.dataset import load_authorized_images, read_image


def create_image(path):
    # Salva uma imagem preta de 10x10 pixels no caminho indicado.
    path.parent.mkdir(parents=True, exist_ok=True)
    _, buffer = cv2.imencode(".png", np.zeros((10, 10, 3), dtype=np.uint8))
    buffer.tofile(path)


def test_subfolder_name_is_the_person_name(tmp_path):
    create_image(tmp_path / "maria" / "1.png")
    create_image(tmp_path / "maria" / "2.png")
    create_image(tmp_path / "joao.png")

    images = load_authorized_images(tmp_path)

    assert sorted(images["name"]) == ["joao", "maria", "maria"]


def test_unsupported_and_hidden_files_are_ignored(tmp_path):
    create_image(tmp_path / "maria" / "1.png")
    (tmp_path / "maria" / "foto.heic").touch()
    (tmp_path / ".DS_Store").touch()
    (tmp_path / ".gitkeep").touch()

    images = load_authorized_images(tmp_path)

    assert images["image_path"].tolist() == [tmp_path / "maria" / "1.png"]


def test_empty_folder_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_authorized_images(tmp_path)


def test_read_image_accepts_accented_paths(tmp_path):
    path = tmp_path / "joão" / "1.png"
    create_image(path)

    image = read_image(path)

    assert image.shape == (10, 10, 3)


def test_authorized_images_are_readable():
    # Confere as fotos reais de data/raw/authorized, se já existirem.
    try:
        images = load_authorized_images(AUTHORIZED_DIR)
    except FileNotFoundError:
        pytest.skip("Nenhuma foto em data/raw/authorized ainda.")

    for image_path in images["image_path"]:
        assert read_image(image_path).ndim == 3
