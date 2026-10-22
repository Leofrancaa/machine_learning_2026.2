"""Scripts para baixar ou gerar os dados do projeto."""

from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from module_deep_learning.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

# Cria a interface de linha de comando (CLI) do script
app = typer.Typer()


@app.command()
def main(
    # Caminhos padrão de entrada e saída (podem ser trocados pela linha de comando)
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
):
    # Estrutura base do template: será substituída pelo código do exercício
    logger.info("Processando o dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Etapa intermediária alcançada (iteração 5).")
    logger.success("Processamento do dataset concluído.")


if __name__ == "__main__":
    app()
