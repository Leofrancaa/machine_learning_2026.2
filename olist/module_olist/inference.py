"""Small inference preview using rows from the intermediate Olist dataset."""

from pathlib import Path

from loguru import logger
import pandas as pd

from module_olist.config import INTERIM_DATA_DIR, MODELS_DIR
from module_olist.modeling.predict import load_model_artifact, predict
from module_olist.modeling.split import split_data

DATASET_PATH = INTERIM_DATA_DIR / "dataset.csv"
MODEL_PATH = MODELS_DIR / "model.pkl"
PREVIEW_ROWS = 5


def run_preview(
    dataset_path: Path = DATASET_PATH,
    model_path: Path = MODEL_PATH,
    rows: int = PREVIEW_ROWS,
) -> pd.DataFrame:
    """Predict delayed orders from the held-out test set."""
    if rows < 1:
        raise ValueError("rows must be at least 1.")
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Run module_olist.main first."
        )

    data = pd.read_csv(dataset_path)
    if data.empty:
        raise ValueError("The inference dataset is empty.")
    if "is_late" not in data:
        raise ValueError("The preview dataset must contain the is_late target.")

    _, _, _, y_test = split_data(data)
    late_test_indices = y_test[y_test.eq(1)].index
    sample = data.loc[late_test_indices].head(rows).copy()
    if sample.empty:
        raise ValueError("No delayed orders were found in the held-out test set.")

    artifact = load_model_artifact(model_path)
    predictions = predict(sample, artifact)

    identity_columns = [column for column in ("order_id", "is_late") if column in sample]
    preview = pd.concat(
        [sample[identity_columns].reset_index(drop=True), predictions.reset_index(drop=True)],
        axis=1,
    )
    preview["late_probability"] = preview["late_probability"].round(4)

    logger.info(
        "Delayed test-order preview with model={} and threshold={:.2f}",
        artifact["model_name"],
        artifact["threshold"],
    )
    print(preview.to_string(index=False))
    return preview


def main() -> None:
    """Display five delayed test-order predictions using default paths."""
    run_preview()


if __name__ == "__main__":
    main()
