"""Curvas de loss e acurácia do treino, salvas em reports/figures."""

from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd
import typer

from module_deep_learning.config import FIGURES_DIR, REPORTS_DIR

# Cria a interface de linha de comando (CLI) do script
app = typer.Typer()


def plot_training_history(history: pd.DataFrame, output_path: Path) -> None:
    """Desenha loss e acurácia de treino e validação por época."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, metric, title in [
        (axes[0], "loss", "Loss (Binary Cross-Entropy)"),
        (axes[1], "accuracy", "Acurácia"),
    ]:
        ax.plot(history[metric], label="treino")
        ax.plot(history[f"val_{metric}"], label="validação")
        ax.set_title(title)
        ax.set_xlabel("Época")
        ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


@app.command()
def main(
    # Caminhos padrão de entrada e saída (podem ser trocados pela linha de comando)
    input_path: Path = REPORTS_DIR / "training_history.csv",
    output_path: Path = FIGURES_DIR / "training_curves.png",
):
    logger.info("Gerando as curvas de treino...")
    history = pd.read_csv(input_path)
    plot_training_history(history, output_path)
    logger.success(f"Gráfico salvo em {output_path}.")


if __name__ == "__main__":
    app()
