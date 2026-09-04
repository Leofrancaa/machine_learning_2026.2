from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from module_olist.inference import run_preview
from module_olist.modeling.predict import load_model_artifact, predict


class ProbabilityModel:
    """Minimal picklable model used to test the inference contract."""

    def predict_proba(self, features):
        probabilities = features["score"].to_numpy(dtype=float)
        return np.column_stack((1 - probabilities, probabilities))


def _save_artifact(path: Path) -> None:
    artifact = {
        "model_name": "test_model",
        "model": ProbabilityModel(),
        "threshold": 0.6,
        "features": ["score"],
    }
    with path.open("wb") as model_file:
        pickle.dump(artifact, model_file)


def test_predict_uses_threshold_stored_in_artifact(tmp_path):
    model_path = tmp_path / "model.pkl"
    _save_artifact(model_path)
    artifact = load_model_artifact(model_path)

    result = predict(pd.DataFrame({"score": [0.2, 0.6, 0.9]}), artifact)

    assert result["late_probability"].tolist() == [0.2, 0.6, 0.9]
    assert result["predicted_is_late"].tolist() == [0, 1, 1]


def test_inference_preview_is_short_and_includes_actual_target(tmp_path):
    model_path = tmp_path / "model.pkl"
    dataset_path = tmp_path / "dataset.csv"
    _save_artifact(model_path)
    row_count = 20
    pd.DataFrame(
        {
            "order_id": [f"o{index}" for index in range(row_count)],
            "score": np.linspace(0.1, 0.9, row_count),
            "is_late": [0, 1] * (row_count // 2),
            "purchase_hour": [12] * row_count,
            "purchase_weekday": [1] * row_count,
            "purchase_month": [6] * row_count,
            "promised_days": [10.0] * row_count,
            "item_count": [1] * row_count,
            "seller_count": [1] * row_count,
            "total_price": [100.0] * row_count,
            "total_freight": [15.0] * row_count,
            "customer_state": ["SP"] * row_count,
        }
    ).to_csv(dataset_path, index=False)

    preview = run_preview(dataset_path, model_path, rows=2)

    assert preview.columns.tolist() == [
        "order_id",
        "is_late",
        "late_probability",
        "predicted_is_late",
    ]
    assert len(preview) == 2
    assert preview["is_late"].eq(1).all()
