"""Testes do treino, da avaliação e dos gráficos (com dados pequenos e falsos)."""

import json

import numpy as np
import pandas as pd
import pytest

from module_classificador_imagens.dataset import CLASS_NAMES
from module_classificador_imagens.modeling.evaluate import evaluate_model
from module_classificador_imagens.modeling.pipeline import INPUT_SHAPE, MODEL_NAME
from module_classificador_imagens.modeling.train import train_model
from module_classificador_imagens.plots import plot_sample_images, plot_training_history


@pytest.fixture
def poucas_imagens():
    # 40 imagens aleatórias já normalizadas: suficiente para testar o fluxo
    rng = np.random.default_rng(42)
    X = rng.random((40, *INPUT_SHAPE), dtype=np.float32)
    y = rng.integers(0, len(CLASS_NAMES), size=40)
    return X, y


def test_treino_salva_modelo_metadados_e_historico(poucas_imagens, tmp_path):
    X, y = poucas_imagens

    model, history = train_model(
        X_train=X,
        y_train=y,
        model_path=tmp_path / "cnn_model.keras",
        metadata_path=tmp_path / "metadata.json",
        history_path=tmp_path / "training_history.csv",
        epochs=1,  # uma época só, para o teste ser rápido
        batch_size=16,
    )

    # Os três arquivos precisam existir
    assert (tmp_path / "cnn_model.keras").exists()
    assert (tmp_path / "training_history.csv").exists()

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["model_name"] == MODEL_NAME
    assert metadata["class_names"] == CLASS_NAMES
    assert metadata["epochs_trained"] == 1

    # O histórico tem uma linha por época com as métricas de validação
    history_df = pd.read_csv(tmp_path / "training_history.csv")
    assert len(history_df) == 1
    assert "val_accuracy" in history_df.columns

    # A avaliação devolve uma acurácia entre 0 e 1
    resultado = evaluate_model(model, MODEL_NAME, X, y)
    assert 0.0 <= resultado["accuracy"] <= 1.0


def test_graficos_sao_salvos_em_png(poucas_imagens, tmp_path):
    X, y = poucas_imagens

    plot_sample_images(X, y, CLASS_NAMES, tmp_path / "sample_images.png")

    history = {
        "accuracy": [0.3, 0.5],
        "val_accuracy": [0.35, 0.45],
        "loss": [1.9, 1.4],
        "val_loss": [1.7, 1.5],
    }
    plot_training_history(history, tmp_path / "training_history.png")

    assert (tmp_path / "sample_images.png").exists()
    assert (tmp_path / "training_history.png").exists()
