"""Pré-processamento das features: normalização das numéricas e one-hot das categóricas."""

import json
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from module_deep_learning.config import MODELS_DIR, PROCESSED_DATA_DIR

# Nome da coluna alvo (1 = tem doença cardíaca, 0 = não tem)
TARGET_FEATURE_NAME = "target"

# Features numéricas: são normalizadas com a média e o desvio padrão do treino
NUMERIC_FEATURE_NAMES = ["age", "trestbps", "thalach", "oldpeak", "slope", "chol"]

# Features categóricas: viram vetores one-hot
CATEGORICAL_FEATURE_NAMES = ["sex", "cp", "fbs", "restecg", "exang", "ca", "thal"]

# Cria a interface de linha de comando (CLI) do script
app = typer.Typer()


def fit_preprocessing(train_dataframe: pd.DataFrame) -> dict:
    """Aprende, só com o treino, as estatísticas e os vocabulários do pré-processamento."""
    numeric = {
        name: {
            "mean": float(train_dataframe[name].mean()),
            # Desvio 0 viraria divisão por zero; nesse caso usa 1
            "std": float(train_dataframe[name].std()) or 1.0,
        }
        for name in NUMERIC_FEATURE_NAMES
    }

    # Categorias guardadas como texto, para tratar números e strings do mesmo jeito
    vocabulary = {
        name: sorted(train_dataframe[name].astype(str).unique().tolist())
        for name in CATEGORICAL_FEATURE_NAMES
    }
    return {"numeric": numeric, "vocabulary": vocabulary}


def encode_features(dataframe: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Transforma as features brutas na matriz numérica que entra na rede."""
    columns = {}

    # Numéricas: (valor - média) / desvio padrão
    for name, stats in params["numeric"].items():
        columns[name] = (dataframe[name].astype(float) - stats["mean"]) / stats["std"]

    # Categóricas: uma coluna 0/1 por categoria conhecida.
    # Categoria que não apareceu no treino fica com todas as colunas em 0.
    for name, categories in params["vocabulary"].items():
        values = dataframe[name].astype(str)
        for category in categories:
            columns[f"{name}_{category}"] = (values == category).astype(np.float32)

    return pd.DataFrame(columns, index=dataframe.index).astype(np.float32)


def save_params(params: dict, path: Path) -> None:
    """Salva os parâmetros do pré-processamento para reutilizar na inferência."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")


def load_params(path: Path) -> dict:
    """Carrega os parâmetros salvos por save_params."""
    return json.loads(path.read_text(encoding="utf-8"))


@app.command()
def main(
    # Caminhos padrão de entrada e saída (podem ser trocados pela linha de comando)
    train_path: Path = PROCESSED_DATA_DIR / "train.csv",
    val_path: Path = PROCESSED_DATA_DIR / "val.csv",
    train_features_path: Path = PROCESSED_DATA_DIR / "train_features.csv",
    val_features_path: Path = PROCESSED_DATA_DIR / "val_features.csv",
    params_path: Path = MODELS_DIR / "preprocessing.json",
):
    logger.info("Gerando as features de treino e validação...")
    train_dataframe = pd.read_csv(train_path)
    val_dataframe = pd.read_csv(val_path)

    # As estatísticas vêm só do treino, para não vazar informação da validação
    params = fit_preprocessing(train_dataframe)
    save_params(params, params_path)

    # Salva as features já codificadas junto com o alvo
    for dataframe, output_path in [
        (train_dataframe, train_features_path),
        (val_dataframe, val_features_path),
    ]:
        features = encode_features(dataframe, params)
        features[TARGET_FEATURE_NAME] = dataframe[TARGET_FEATURE_NAME]
        features.to_csv(output_path, index=False)

    logger.success(f"Features geradas: {features.shape[1] - 1} colunas de entrada.")


if __name__ == "__main__":
    app()
