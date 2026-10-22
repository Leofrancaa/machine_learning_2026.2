"""Carregamento da base CIFAR-10."""

from loguru import logger
import numpy as np
from tensorflow import keras

# Nome de cada classe do CIFAR-10. A posição na lista é o rótulo:
# rótulo 0 = "avião", rótulo 1 = "automóvel", ..., rótulo 9 = "caminhão".
CLASS_NAMES = [
    "avião",
    "automóvel",
    "pássaro",
    "gato",
    "cervo",
    "cachorro",
    "sapo",
    "cavalo",
    "navio",
    "caminhão",
]


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Carrega as imagens e os rótulos do CIFAR-10.

    Na primeira vez o Keras baixa a base (cerca de 170 MB) para a pasta
    ~/.keras/datasets. Depois disso, ela é lida direto do disco.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
            - X_train: 50.000 imagens de treino, formato (50000, 32, 32, 3).
            - X_test: 10.000 imagens de teste, formato (10000, 32, 32, 3).
            - y_train: rótulos de treino, formato (50000,).
            - y_test: rótulos de teste, formato (10000,).
    """
    logger.info("Carregando a base CIFAR-10...")

    # O Keras já entrega a base separada em treino e teste
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

    # Os rótulos vêm como coluna (n, 1). O flatten transforma em vetor (n,),
    # o que facilita usar CLASS_NAMES[y[i]] e comparar com as predições.
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    logger.info(f"Treino: {X_train.shape[0]} imagens | Teste: {X_test.shape[0]} imagens")
    logger.info(f"Formato de cada imagem: {X_train.shape[1:]}")

    return X_train, X_test, y_train, y_test
