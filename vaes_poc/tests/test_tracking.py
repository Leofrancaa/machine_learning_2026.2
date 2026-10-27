import mlflow

from module_vae import tracking
from module_vae.tracking import flatten


def test_flatten_nested_metrics_and_drops_non_numbers():
    metrics = {"auc": 0.75, "ranking": {"precision_at_10": 0.4}, "name": "x", "ok": True}

    assert flatten(metrics, "val") == {"val_auc": 0.75, "val_ranking_precision_at_10": 0.4}


def test_new_experiments_open_as_model_training(tmp_path, monkeypatch):
    monkeypatch.setattr(tracking, "MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setattr(tracking, "MLFLOW_ARTIFACTS_DIR", tmp_path / "mlartifacts")

    experiment_id = tracking.get_experiment_id("teste")

    tags = mlflow.get_experiment(experiment_id).tags
    assert tags["mlflow.experimentKind"] == "custom_model_development"
    assert tracking.get_experiment_id("teste") == experiment_id  # não cria de novo
