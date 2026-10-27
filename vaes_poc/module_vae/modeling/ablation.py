"""Experimentos com hiperparâmetros.

Varia o tamanho do espaço latente e o peso β do KL, um de cada vez, e mede a AUC na
validação (o teste não é usado para escolher nada). Cada configuração é treinada uma vez
(semente 0), então diferenças pequenas (< 0,02) estão dentro da variação entre sementes.

Usa as incidências, a resolução e o σ salvos nos metadados da execução, então rode
`make train` (ou o main.py com as mesmas opções) antes.
"""

import json
from typing import Annotated

from loguru import logger
import pandas as pd
from sklearn.metrics import roc_auc_score
import typer

from module_vae.config import BETA, RunPaths
from module_vae.features import anomaly_score
from module_vae.modeling.predict import reconstruct
from module_vae.modeling.split import load_splits
from module_vae.modeling.train import train_autoencoder
from module_vae.plots import apply_style, plot_ablation

LATENT_DIMS = [2, 4, 8, 16, 32]
BETAS = [0.25, 0.5, 1.0, 2.0, 4.0]

app = typer.Typer()


@app.command()
def main(
    run: Annotated[
        str, typer.Option(help="Execução treinada: rsna_pa, rsna_ap_pa, rsna_pa_v2...")
    ] = "rsna_pa",
):
    """Treina cada configuração e registra a AUC de validação."""
    paths = RunPaths(run)
    if not paths.metadata.exists():
        raise FileNotFoundError(f"{paths.metadata} não existe. Rode `make train` antes.")
    with open(paths.metadata, "r", encoding="utf-8") as file:
        metadata = json.load(file)
    sigma, epochs, base_latent = metadata["sigma"], metadata["epochs"], metadata["latent_dim"]
    logger.info(f"Execução {run} | σ: {sigma} | épocas: {epochs}")

    splits = load_splits(
        metadata["image_size"],
        metadata["views"],
        # modelos da v1 não têm "hygiene" nos metadados: foram treinados sem higiene
        hygiene=metadata.get("hygiene", False),
    )

    # A configuração da execução aparece nas duas varreduras; ela é treinada uma vez só.
    configs = [(d, BETA) for d in LATENT_DIMS] + [(base_latent, b) for b in BETAS if b != BETA]

    rows = []
    for latent_dim, beta in configs:
        logger.info(f"Treinando VAE com latente {latent_dim} e β {beta}...")
        model, _ = train_autoencoder(
            splits.x_train_normal,
            splits.x_val_normal,
            seed=0,
            latent_dim=latent_dim,
            beta=beta,
            epochs=epochs,
            verbose=False,
        )
        score_val = anomaly_score(splits.x_val, reconstruct(model, splits.x_val), sigma)
        rows.append(
            {
                "latent_dim": latent_dim,
                "beta": beta,
                "auc_val": roc_auc_score(splits.y_val, score_val),
            }
        )
        logger.info(f"AUC (validação): {rows[-1]['auc_val']:.3f}")

    history = pd.DataFrame(rows)
    paths.reports.mkdir(parents=True, exist_ok=True)
    history.to_csv(paths.reports / "ablation.csv", index=False)
    logger.success(f"Resultados salvos em {paths.reports / 'ablation.csv'}")

    apply_style()
    plot_ablation(
        latent=history[history["beta"] == BETA].set_index("latent_dim")["auc_val"],
        beta=history[history["latent_dim"] == base_latent]
        .set_index("beta")["auc_val"]
        .sort_index(),
        path=paths.figures / "ablation.png",
    )


if __name__ == "__main__":
    app()
