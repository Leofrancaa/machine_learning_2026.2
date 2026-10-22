"""Código para criar as features usadas na modelagem."""

from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from module_deep_learning.config import PROCESSED_DATA_DIR

# Cria a interface de linha de comando (CLI) do script
app = typer.Typer()


@app.command()
def main(
    # Caminhos padrão de entrada e saída (podem ser trocados pela linha de comando)
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
):
    # Estrutura base do template: será substituída pelo código do exercício
    logger.info("Gerando features a partir do dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Etapa intermediária alcançada (iteração 5).")
    logger.success("Geração de features concluída.")


if __name__ == "__main__":
    app()
