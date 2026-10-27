"""Inferência com a rede treinada: probabilidade de doença cardíaca por paciente."""

from pathlib import Path

import keras
from loguru import logger
import pandas as pd
import typer

from module_deep_learning.config import MODELS_DIR, PROCESSED_DATA_DIR
from module_deep_learning.features import (
    TARGET_FEATURE_NAME,
    encode_features,
    load_params,
)

# Probabilidade a partir da qual o paciente é classificado como "com doença"
THRESHOLD = 0.5

# Paciente de exemplo do notebook
EXAMPLE_PATIENT = {
    "age": 60,
    "sex": 1,
    "cp": 1,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 2,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 3,
    "ca": 0,
    "thal": "fixed",
}

# Cria a interface de linha de comando (CLI) do script
app = typer.Typer()


def predict_proba(model: keras.Model, dataframe: pd.DataFrame, params: dict) -> pd.Series:
    """Recebe pacientes com as features brutas e devolve a probabilidade de cada um."""
    features = encode_features(dataframe, params)
    probabilities = model.predict(features.to_numpy("float32"), verbose=0)
    return pd.Series(probabilities[:, 0], index=dataframe.index, name="probability")


@app.command()
def main(
    # Caminhos padrão de entrada e saída (podem ser trocados pela linha de comando)
    input_path: Path = PROCESSED_DATA_DIR / "val.csv",
    model_path: Path = MODELS_DIR / "model.keras",
    params_path: Path = MODELS_DIR / "preprocessing.json",
    predictions_path: Path = PROCESSED_DATA_DIR / "val_predictions.csv",
):
    logger.info("Executando a inferência do modelo...")
    model = keras.models.load_model(model_path)
    params = load_params(params_path)

    # Previsões para os pacientes do arquivo de entrada
    patients = pd.read_csv(input_path)
    result = patients.copy()
    result["probability"] = predict_proba(model, patients, params)
    result["prediction"] = (result["probability"] >= THRESHOLD).astype(int)
    result.to_csv(predictions_path, index=False)

    # Se o arquivo tiver o alvo, mostra a acurácia das previsões
    if TARGET_FEATURE_NAME in result:
        accuracy = (result["prediction"] == result[TARGET_FEATURE_NAME]).mean()
        logger.info(f"Acurácia em {input_path.name}: {accuracy:.3f}")

    # Previsão para o paciente de exemplo
    example = pd.DataFrame([EXAMPLE_PATIENT])
    probability = predict_proba(model, example, params).iloc[0]
    logger.info(
        f"Paciente de exemplo: {100 * probability:.1f}% de probabilidade "
        "de ter doença cardíaca, segundo o modelo."
    )

    logger.success(f"Previsões salvas em {predictions_path}.")


if __name__ == "__main__":
    app()
