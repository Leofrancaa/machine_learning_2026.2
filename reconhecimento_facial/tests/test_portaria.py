import base64

import cv2
import numpy as np
import pytest

from module_reconhecimento.portaria import bytes_to_image, clean_name, decode_data_url


def test_clean_name_removes_extra_spaces():
    assert clean_name("  Maria   Souza ") == "Maria Souza"
    assert clean_name("João") == "João"


@pytest.mark.parametrize("name", ["", "../joao", "maria/1", "a" * 41])
def test_clean_name_rejects_invalid_names(name):
    # Nomes que sairiam da pasta authorized/ ou não viram nome de pasta.
    with pytest.raises(ValueError):
        clean_name(name)


def test_photo_sent_by_the_page_is_decoded():
    _, buffer = cv2.imencode(".png", np.zeros((10, 20, 3), dtype=np.uint8))
    data_url = "data:image/png;base64," + base64.b64encode(buffer.tobytes()).decode()

    raw, extension = decode_data_url(data_url)

    assert extension == ".png"
    assert bytes_to_image(raw).shape == (10, 20, 3)


def test_unsupported_format_is_rejected():
    with pytest.raises(ValueError):
        decode_data_url("data:image/heic;base64,AAAA")
