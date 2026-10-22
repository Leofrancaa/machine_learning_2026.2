"""Código para fazer inferência com os modelos treinados."""

from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from module_deep_learning.config import MODELS_DIR, PROCESSED_DATA_DIR

# Cria a interface de linha de comando (CLI) do script
app = typer.Typer()


@app.command()
def main(
    # Caminhos padrão de entrada e saída (podem ser trocados pela linha de comando)
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model.keras",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
):
    # Estrutura base do template: será substituída pelo código do exercício
    logger.info("Executando a inferência do modelo...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Etapa intermediária alcançada (iteração 5).")
    logger.success("Inferência concluída.")


if __name__ == "__main__":
    app()
