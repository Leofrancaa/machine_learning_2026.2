"""Registro das execuções no MLflow.

O rastreamento é local: parâmetros e métricas ficam em mlflow.db e os artefatos (figuras,
tabelas, pesos) em mlartifacts/, os dois fora do git. Para abrir a interface: `make mlflow`.

Convenção dos nomes das métricas:
- train_* e val_* acompanham o treino e a calibração. As val_* são as usadas para escolher
  entre configurações.
- test_* só devem ser olhadas na configuração final, para o teste não virar critério de
  escolha.
"""

from collections.abc import Iterator
from contextlib import contextmanager
import subprocess

import mlflow
import pandas as pd

from module_vae.config import MLFLOW_ARTIFACTS_DIR, MLFLOW_TRACKING_URI, PROJ_ROOT

# Sem essa tag, o MLflow 3 abre o experimento na visão de aplicações com LLM (traces,
# judges); com ela, abre direto na lista de execuções de treino.
EXPERIMENT_KIND = {"mlflow.experimentKind": "custom_model_development"}


def get_experiment_id(name: str) -> str:
    """Devolve o id do experimento, criando-o (com artefatos em mlartifacts/) se preciso."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id
    return mlflow.create_experiment(
        name,
        artifact_location=(MLFLOW_ARTIFACTS_DIR / name).as_uri(),
        tags=EXPERIMENT_KIND,
    )


def git_commit() -> str:
    """Commit atual do repositório, com "-dirty" se houver alterações não commitadas."""

    def git(*args: str) -> str:
        # check=False: sem git (ou fora de um repositório), a execução segue sem o commit
        result = subprocess.run(
            ["git", *args], cwd=PROJ_ROOT, capture_output=True, text=True, check=False
        )
        return result.stdout.strip()

    commit = git("rev-parse", "--short", "HEAD")
    return commit + ("-dirty" if git("status", "--porcelain", "--untracked-files=no") else "")


@contextmanager
def tracked_run(
    experiment: str, run_name: str, params: dict, tags: dict | None = None
) -> Iterator[mlflow.ActiveRun]:
    """Abre uma execução no MLflow com os parâmetros e o commit do código que a gerou."""
    with mlflow.start_run(experiment_id=get_experiment_id(experiment), run_name=run_name) as run:
        mlflow.set_tags({"git_commit": git_commit(), **(tags or {})})
        mlflow.log_params(params)
        yield run


def flatten(metrics: dict, prefix: str) -> dict[str, float]:
    """Achata dicionários aninhados em métricas: {"a": {"b": 1}} → {"<prefix>_a_b": 1.0}."""
    flat = {}
    for key, value in metrics.items():
        name = f"{prefix}_{key}"
        if isinstance(value, dict):
            flat |= flatten(value, name)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            flat[name] = float(value)
    return flat


def log_history(history: dict[str, list[float]]) -> None:
    """Curvas de treino, uma métrica por época: train_rec, train_kl, val_rec e val_kl."""
    for key, values in history.items():
        term, split = key.split("_")  # "rec_train" → train_rec
        for epoch, value in enumerate(values, start=1):
            mlflow.log_metric(f"{split}_{term}", value, step=epoch)


def log_comparison(table: pd.DataFrame) -> None:
    """Uma execução filha por método da comparação, para comparar os métodos na interface."""
    names = {"auc_val": "val_auc", "auc_test": "test_auc", "auc_test_std": "test_auc_std",
             "ap_test": "test_ap"}  # fmt: skip
    # As filhas vão para o mesmo experimento da execução principal (sem isso, cairiam no
    # experimento "Default").
    experiment_id = mlflow.active_run().info.experiment_id
    for method, row in table.iterrows():
        with mlflow.start_run(run_name=str(method), nested=True, experiment_id=experiment_id):
            mlflow.log_params(
                {
                    "method": method,
                    "uses_labels": bool(row["uses_labels"]),
                    "sigma": row["sigma"],
                    "n_runs": int(row["n_runs"]),
                }
            )
            mlflow.log_metrics(
                {new: float(row[old]) for old, new in names.items() if pd.notna(row[old])}
            )
