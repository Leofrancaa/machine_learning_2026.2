"""Download do dataset de doença cardíaca e divisão em treino e validação."""

from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from module_deep_learning.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

# Mesmo CSV usado no notebook structured_data_classification_from_scratch.ipynb
DATA_URL = "http://storage.googleapis.com/download.tensorflow.org/data/heart.csv"

# Fração das linhas que vai para validação e semente fixa para a divisão ser reproduzível
VAL_FRACTION = 0.2
RANDOM_STATE = 1337

# Cria a interface de linha de comando (CLI) do script
app = typer.Typer()


def load_data(url: str = DATA_URL) -> pd.DataFrame:
    """Lê o CSV de pacientes (303 linhas, 13 features e a coluna alvo `target`)."""
    return pd.read_csv(url)


def split_data(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa 20% das linhas para validação e deixa o restante para treino."""
    val_dataframe = dataframe.sample(frac=VAL_FRACTION, random_state=RANDOM_STATE)
    train_dataframe = dataframe.drop(val_dataframe.index)
    return train_dataframe, val_dataframe


@app.command()
def main(
    # Caminhos padrão de saída (podem ser trocados pela linha de comando)
    raw_path: Path = RAW_DATA_DIR / "heart.csv",
    train_path: Path = PROCESSED_DATA_DIR / "train.csv",
    val_path: Path = PROCESSED_DATA_DIR / "val.csv",
):
    logger.info(f"Baixando o dataset de {DATA_URL}...")
    dataframe = load_data()

    # Guarda uma cópia do dado original em data/raw
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(raw_path, index=False)

    train_dataframe, val_dataframe = split_data(dataframe)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_dataframe.to_csv(train_path, index=False)
    val_dataframe.to_csv(val_path, index=False)

    logger.success(
        f"Dataset salvo: {len(train_dataframe)} amostras de treino "
        f"e {len(val_dataframe)} de validação."
    )


if __name__ == "__main__":
    app()
