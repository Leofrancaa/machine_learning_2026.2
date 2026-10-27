"""Treino da rede neural (Keras) que classifica doença cardíaca."""

from pathlib import Path

import keras
from keras import layers
from loguru import logger
import pandas as pd
import typer

from module_deep_learning.config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from module_deep_learning.features import TARGET_FEATURE_NAME

EPOCHS = 50
BATCH_SIZE = 32
SEED = 1337

# Cria a interface de linha de comando (CLI) do script
app = typer.Typer()


def split_features_target(dataframe: pd.DataFrame):
    """Separa a matriz de features (X) da coluna alvo (y)."""
    X = dataframe.drop(columns=[TARGET_FEATURE_NAME]).to_numpy("float32")
    y = dataframe[TARGET_FEATURE_NAME].to_numpy("float32")
    return X, y


def create_model(num_features: int) -> keras.Model:
    """Mesma arquitetura do notebook: Dense(32, ReLU) -> Dropout(0.5) -> Dense(1, sigmoid)."""
    model = keras.Sequential(
        [
            keras.Input(shape=(num_features,)),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.5),
            # Sigmoid devolve a probabilidade de o paciente ter doença cardíaca
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    # Adam como otimizador, Binary Cross-Entropy como loss e acurácia como métrica
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_model(
    train_dataframe: pd.DataFrame,
    val_dataframe: pd.DataFrame,
    epochs: int = EPOCHS,
) -> tuple[keras.Model, pd.DataFrame]:
    """Treina a rede e devolve o modelo e o histórico (loss e acurácia por época)."""
    # Semente fixa para o treino ser reproduzível
    keras.utils.set_random_seed(SEED)

    X_train, y_train = split_features_target(train_dataframe)
    X_val, y_val = split_features_target(val_dataframe)

    model = create_model(X_train.shape[1])
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        verbose=2,
    )

    history_dataframe = pd.DataFrame(history.history)
    history_dataframe.index.name = "epoch"
    return model, history_dataframe


@app.command()
def main(
    # Caminhos padrão de entrada e saída (podem ser trocados pela linha de comando)
    train_features_path: Path = PROCESSED_DATA_DIR / "train_features.csv",
    val_features_path: Path = PROCESSED_DATA_DIR / "val_features.csv",
    model_path: Path = MODELS_DIR / "model.keras",
    history_path: Path = REPORTS_DIR / "training_history.csv",
    epochs: int = EPOCHS,
):
    logger.info("Treinando a rede neural...")
    train_dataframe = pd.read_csv(train_features_path)
    val_dataframe = pd.read_csv(val_features_path)

    model, history = train_model(train_dataframe, val_dataframe, epochs=epochs)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(history_path)

    last = history.iloc[-1]
    logger.success(
        f"Modelo salvo em {model_path.name}. Acurácia final: "
        f"treino {last['accuracy']:.3f} | validação {last['val_accuracy']:.3f}"
    )


if __name__ == "__main__":
    app()
