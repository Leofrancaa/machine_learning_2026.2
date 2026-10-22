"""Carregamento do modelo treinado e predição de novas imagens."""

import json

from loguru import logger
import numpy as np
import pandas as pd
from tensorflow import keras


def load_model(model_path, metadata_path):
    """
    Carrega o modelo salvo no treino e os metadados dele.

    Args:
        model_path (Path): arquivo do modelo (.keras).
        metadata_path (Path): arquivo de metadados (.json).

    Returns:
        tuple: (modelo, nome do modelo, nomes das classes)
    """
    model = keras.models.load_model(model_path)

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    model_name = metadata["model_name"]
    class_names = metadata["class_names"]

    logger.info(f"Modelo carregado: {model_name}")
    logger.info(f"Classes: {class_names}")

    return model, model_name, class_names


def predict(model, X, class_names) -> pd.DataFrame:
    """
    Prevê a classe de cada imagem.

    Args:
        model (keras.Model): modelo treinado.
        X (np.ndarray): imagens já normalizadas, formato (n, 32, 32, 3).
        class_names (list[str]): nome de cada classe.

    Returns:
        pd.DataFrame: uma linha por imagem com o rótulo previsto (prediction),
            o nome da classe (class_name) e a probabilidade dela (confidence).
    """
    # Probabilidade de cada uma das 10 classes para cada imagem: formato (n, 10)
    y_proba = model.predict(X, verbose=0)

    # A classe prevista é a de maior probabilidade
    y_pred = np.argmax(y_proba, axis=1)

    return pd.DataFrame(
        {
            "prediction": y_pred,
            "class_name": [class_names[i] for i in y_pred],
            "confidence": y_proba.max(axis=1),
        }
    )
