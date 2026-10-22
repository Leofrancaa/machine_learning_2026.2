"""Preparação das imagens antes de entrar na rede."""

import numpy as np


def normalize_images(X: np.ndarray) -> np.ndarray:
    """
    Coloca os pixels das imagens no intervalo [0, 1].

    Args:
        X (np.ndarray): imagens com pixels de 0 a 255 (uint8).

    Returns:
        np.ndarray: imagens em float32 com pixels de 0 a 1.
    """
    # Cada pixel vai de 0 a 255. Dividindo por 255, todos ficam entre 0 e 1,
    # escala em que a rede neural treina de forma mais estável.
    return X.astype("float32") / 255.0
