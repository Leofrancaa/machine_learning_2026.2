"""Preview predictions from the selected Olist model."""

import pandas as pd

from module_olist.config import INTERIM_DATA_DIR, MODELS_DIR
from module_olist.modeling.predict import load_model, predict_with_threshold
from module_olist.modeling.split import FEATURES


def main() -> None:
    dataset_path = INTERIM_DATA_DIR / "dataset.csv"
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Prepared dataset not found at {dataset_path}.")
    data = pd.read_csv(dataset_path)
    model, model_name, threshold = load_model(
        MODELS_DIR / "best_model.joblib", MODELS_DIR / "metadata.json"
    )
    sample = data[FEATURES].sample(n=min(5, len(data)), random_state=42)
    print(f"Model: {model_name}; threshold: {threshold:.2f}")
    print(predict_with_threshold(model, sample, threshold).to_string(index=False))


if __name__ == "__main__":
    main()
