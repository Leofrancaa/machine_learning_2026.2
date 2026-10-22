"""Treino da CNN e gravação do modelo, dos metadados e do histórico."""

import json

from loguru import logger
import pandas as pd
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from module_classificador_imagens.dataset import CLASS_NAMES
from module_classificador_imagens.modeling.pipeline import (
    INPUT_SHAPE,
    MODEL_NAME,
    create_cnn_model,
)

# Semente para o treino ser reproduzível
RANDOM_STATE = 42

# Configuração do treino (a mesma usada no notebook)
EPOCHS = 60
BATCH_SIZE = 64

# O Keras separa os últimos 20% do treino para validação
VALIDATION_SPLIT = 0.2


def create_callbacks() -> list:
    """
    Cria os callbacks que controlam o treino.

    Returns:
        list: [EarlyStopping, ReduceLROnPlateau]
    """
    # Para o treino se a val_loss não melhorar por 8 épocas
    # e volta para os pesos da melhor época
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
    )

    # Corta a learning rate pela metade se a val_loss ficar parada por 3 épocas
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-5,
    )

    return [early_stopping, reduce_lr]


def train_model(
    X_train,
    y_train,
    model_path,
    metadata_path,
    history_path,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
):
    """
    Treina a CNN e salva o modelo, os metadados e o histórico do treino.

    Args:
        X_train (np.ndarray): imagens de treino já normalizadas.
        y_train (np.ndarray): rótulos de treino.
        model_path (Path): onde salvar o modelo (.keras).
        metadata_path (Path): onde salvar os metadados (.json).
        history_path (Path): onde salvar o histórico por época (.csv).
        epochs (int): número máximo de épocas.
        batch_size (int): quantidade de imagens por lote.

    Returns:
        tuple: (modelo treinado, histórico do fit)
    """
    logger.info(f"Treinando o modelo {MODEL_NAME}...")

    # Fixa as sementes do Python, NumPy e TensorFlow (pesos iniciais,
    # augmentation e dropout ficam iguais a cada execução)
    keras.utils.set_random_seed(RANDOM_STATE)

    model = create_cnn_model()

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=VALIDATION_SPLIT,
        callbacks=create_callbacks(),
    )

    # ---------- Modelo ----------
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    logger.success(f"Modelo salvo em: {model_path}")

    # ---------- Histórico ----------
    # Uma linha por época (loss, accuracy, val_loss, val_accuracy, learning_rate).
    # Com ele dá para refazer os gráficos sem treinar de novo.
    history_df = pd.DataFrame(history.history)
    history_df.index.name = "epoch"

    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(history_path)
    logger.success(f"Histórico salvo em: {history_path}")

    # ---------- Metadados ----------
    # Informações usadas depois na inferência (nomes das classes, formato da entrada)
    metadata = {
        "model_name": MODEL_NAME,
        "class_names": CLASS_NAMES,
        "input_shape": list(INPUT_SHAPE),
        "epochs_trained": len(history_df),
        "best_val_loss": float(history_df["val_loss"].min()),
    }

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)
    logger.success(f"Metadados salvos em: {metadata_path}")

    return model, history
