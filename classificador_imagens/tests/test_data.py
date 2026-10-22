"""Testes do carregamento e da normalização do CIFAR-10."""

import numpy as np
import pytest

from module_classificador_imagens.dataset import CLASS_NAMES, load_data
from module_classificador_imagens.features import normalize_images

# Tamanhos esperados da base CIFAR-10
N_TRAIN = 50_000
N_TEST = 10_000
IMAGE_SHAPE = (32, 32, 3)


@pytest.fixture(scope="module")
def cifar10():
    # Carrega a base só uma vez e reaproveita em todos os testes do arquivo
    return load_data()


def test_existem_dez_classes():
    assert len(CLASS_NAMES) == 10


def test_imagens_tem_formato_e_tipo_esperados(cifar10):
    X_train, X_test, _, _ = cifar10

    assert X_train.shape == (N_TRAIN, *IMAGE_SHAPE)
    assert X_test.shape == (N_TEST, *IMAGE_SHAPE)
    assert X_train.dtype == np.uint8


def test_rotulos_sao_vetores_com_classes_validas(cifar10):
    _, _, y_train, y_test = cifar10

    # Depois do flatten, os rótulos são vetores (n,), não colunas (n, 1)
    assert y_train.shape == (N_TRAIN,)
    assert y_test.shape == (N_TEST,)

    # Todo rótulo precisa estar entre 0 e 9
    for y in (y_train, y_test):
        assert y.min() >= 0
        assert y.max() < len(CLASS_NAMES)


def test_treino_tem_classes_balanceadas(cifar10):
    _, _, y_train, _ = cifar10

    # O CIFAR-10 tem 5.000 imagens de treino para cada classe
    contagem = np.bincount(y_train, minlength=len(CLASS_NAMES))
    assert (contagem == N_TRAIN // len(CLASS_NAMES)).all()


def test_normalizacao_fica_entre_zero_e_um():
    # Imagem pequena criada à mão, com o menor e o maior valor possível de pixel
    X = np.array([[[[0, 128, 255]]]], dtype=np.uint8)

    X_norm = normalize_images(X)

    assert X_norm.dtype == np.float32
    assert X_norm.min() == 0.0
    assert X_norm.max() == 1.0
