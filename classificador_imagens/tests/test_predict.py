"""Testes da inferência com um modelo treinado rapidamente."""

import numpy as np
import pytest

from module_classificador_imagens.dataset import CLASS_NAMES
from module_classificador_imagens.modeling.pipeline import INPUT_SHAPE
from module_classificador_imagens.modeling.predict import load_model, predict
from module_classificador_imagens.modeling.train import train_model


@pytest.fixture(scope="module")
def modelo_salvo(tmp_path_factory):
    # Treina 1 época com poucas imagens falsas só para ter um modelo em disco
    pasta = tmp_path_factory.mktemp("modelo")
    rng = np.random.default_rng(42)
    X = rng.random((40, *INPUT_SHAPE), dtype=np.float32)
    y = rng.integers(0, len(CLASS_NAMES), size=40)

    train_model(
        X_train=X,
        y_train=y,
        model_path=pasta / "cnn_model.keras",
        metadata_path=pasta / "metadata.json",
        history_path=pasta / "training_history.csv",
        epochs=1,
        batch_size=16,
    )
    return pasta, X


def test_carrega_modelo_e_nomes_das_classes(modelo_salvo):
    pasta, _ = modelo_salvo

    model, model_name, class_names = load_model(
        pasta / "cnn_model.keras", pasta / "metadata.json"
    )

    assert model_name == "CNN"
    assert class_names == CLASS_NAMES


def test_predicao_retorna_uma_linha_por_imagem(modelo_salvo):
    pasta, X = modelo_salvo
    model, _, class_names = load_model(pasta / "cnn_model.keras", pasta / "metadata.json")

    predictions = predict(model, X[:5], class_names)

    assert len(predictions) == 5
    assert list(predictions.columns) == ["prediction", "class_name", "confidence"]

    # Rótulo dentro de 0-9, nome coerente com o rótulo e confiança entre 0 e 1
    assert predictions["prediction"].between(0, 9).all()
    assert all(
        class_names[p] == nome
        for p, nome in zip(predictions["prediction"], predictions["class_name"])
    )
    assert predictions["confidence"].between(0, 1).all()
