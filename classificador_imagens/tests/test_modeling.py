"""Testes da arquitetura da CNN."""

import numpy as np

from module_classificador_imagens.modeling.pipeline import (
    INPUT_SHAPE,
    NUM_CLASSES,
    create_cnn_model,
)


def test_modelo_gera_uma_probabilidade_por_classe():
    model = create_cnn_model()

    # Lote com 4 imagens aleatórias, já no formato normalizado
    X = np.random.default_rng(42).random((4, *INPUT_SHAPE), dtype=np.float32)
    y_proba = model.predict(X, verbose=0)

    # Uma linha por imagem e uma coluna por classe
    assert y_proba.shape == (4, NUM_CLASSES)

    # O softmax faz cada linha somar 1
    assert np.allclose(y_proba.sum(axis=1), 1.0, atol=1e-5)


def test_modelo_tem_a_quantidade_esperada_de_parametros():
    model = create_cnn_model()

    # Mesmo total de parâmetros da CNN do notebook (model.summary)
    assert model.count_params() == 552_362
