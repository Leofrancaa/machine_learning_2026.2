"""Código para gerar as visualizações do projeto."""

from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from module_deep_learning.config import FIGURES_DIR, PROCESSED_DATA_DIR

# Cria a interface de linha de comando (CLI) do script
app = typer.Typer()


@app.command()
def main(
    # Caminhos padrão de entrada e saída (podem ser trocados pela linha de comando)
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
):
    # Estrutura base do template: será substituída pelo código do exercício
    logger.info("Gerando gráfico a partir dos dados...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Etapa intermediária alcançada (iteração 5).")
    logger.success("Geração do gráfico concluída.")


if __name__ == "__main__":
    app()
