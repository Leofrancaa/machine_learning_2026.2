"""Aplica a regra das ablações às execuções registradas no MLflow.

    uv run python -m module_vae.modeling.compare_runs --candidates abl-A-recorte,abl-B-normalizacao

Cada ablação (um nome de execução, com uma execução por semente) é comparada com a v2
(ABLATION_REFERENCE) pela média das sementes, usando só métricas de validação. A regra
está em config.py e foi fixada antes de qualquer ablação rodar; a localização é medida
acima do acaso de cada configuração.
"""

from typing import Annotated

from loguru import logger
import mlflow
import pandas as pd
import typer

from module_vae.config import (
    ABLATION_MIN_AUC_GAIN,
    ABLATION_MIN_POINTING_GAIN,
    ABLATION_REFERENCE,
    MLFLOW_TRACKING_URI,
    REPORTS_DIR,
)
from module_vae.plots import apply_style, plot_ablation_rule

AUC = "metrics.val_auc"
POINTING = "metrics.val_localization_pointing_game"
CHANCE = "metrics.val_localization_box_area_fraction"
CHANCE_IN_MASK = "metrics.val_localization_box_fraction_in_mask"

app = typer.Typer()


def chance_level(runs: pd.DataFrame) -> pd.Series:
    """Localização de um mapa sem informação, por execução.

    É a fração da área onde o mapa existe coberta pelas caixas: a imagem inteira, ou só a
    máscara dos pulmões (ablação C1), onde o acaso é bem maior.
    """
    chance = runs[CHANCE]
    if CHANCE_IN_MASK in runs:
        chance = runs[CHANCE_IN_MASK].fillna(chance)
    return chance


def apply_rule(reference: pd.DataFrame, candidate: pd.DataFrame) -> dict:
    """Compara uma ablação com a referência pela média das sementes.

    A localização entra como "acima do acaso" (acerto − acaso de cada execução). Sem
    isso, a máscara dos pulmões ganharia pontos só por encolher a área onde o mapa
    existe. Correção aprovada em 18/10/2026, antes de qualquer resultado de ablação.

    Args:
        reference (pd.DataFrame): Execuções da v2, uma por semente.
        candidate (pd.DataFrame): Execuções da ablação, uma por semente.

    Returns:
        dict: Médias, diferenças, tolerâncias (um desvio-padrão da referência) e a decisão.
    """
    lift_reference = reference[POINTING] - chance_level(reference)
    lift_candidate = candidate[POINTING] - chance_level(candidate)

    delta_auc = candidate[AUC].mean() - reference[AUC].mean()
    delta_lift = lift_candidate.mean() - lift_reference.mean()
    tolerance_auc, tolerance_lift = reference[AUC].std(), lift_reference.std()

    gain = delta_auc >= ABLATION_MIN_AUC_GAIN or delta_lift >= ABLATION_MIN_POINTING_GAIN
    no_harm = delta_auc >= -tolerance_auc and delta_lift >= -tolerance_lift

    return {
        "n_seeds": len(candidate),
        "val_auc": candidate[AUC].mean(),
        "val_auc_std": candidate[AUC].std(),
        "delta_auc": delta_auc,
        "val_pointing": candidate[POINTING].mean(),
        "chance": chance_level(candidate).mean(),
        "pointing_above_chance": lift_candidate.mean(),
        "pointing_above_chance_std": lift_candidate.std(),
        "delta_above_chance": delta_lift,
        "tolerance_auc": tolerance_auc,
        "tolerance_above_chance": tolerance_lift,
        "accepted": bool(gain and no_harm),
    }


@app.command()
def main(
    candidates: Annotated[str, typer.Option(help="Nomes das ablações, separados por vírgula.")],
    experiment: Annotated[str, typer.Option(help="Experimento no MLflow.")] = "rsna_pa",
    output: Annotated[str, typer.Option(help="Pasta em reports/ para a tabela.")] = (
        "rsna_pa_ablations_v2"
    ),
):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    runs = mlflow.search_runs(
        experiment_names=[experiment], filter_string="attributes.status = 'FINISHED'"
    )
    runs = (
        runs[runs["tags.mlflow.parentRunId"].isna()] if "tags.mlflow.parentRunId" in runs else runs
    )
    by_name = runs.groupby("tags.mlflow.runName")

    reference = pd.concat([by_name.get_group(name) for name in ABLATION_REFERENCE])
    groups = {"v2 (referência)": reference}
    groups |= {name.strip(): by_name.get_group(name.strip()) for name in candidates.split(",")}
    rows = {name: apply_rule(reference, runs) for name, runs in groups.items()}
    rows["v2 (referência)"]["accepted"] = None

    table = pd.DataFrame(rows).T

    path = REPORTS_DIR / output / "ablations.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index_label="run")
    logger.success(f"Regra das ablações (validação, média das sementes):\n{table.to_string()}")
    logger.info(f"Tabela salva em {path}")

    apply_style()
    plot_ablation_rule(
        {
            name: pd.DataFrame(
                {
                    "auc": runs[AUC].to_numpy(),
                    "lift": (runs[POINTING] - chance_level(runs)).to_numpy(),
                }
            )
            for name, runs in groups.items()
        },
        ABLATION_MIN_AUC_GAIN,
        ABLATION_MIN_POINTING_GAIN,
        {name: row["accepted"] for name, row in rows.items()},
        REPORTS_DIR / output / "figures" / "ablations.png",
    )


if __name__ == "__main__":
    app()
