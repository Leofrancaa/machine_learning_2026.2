"""Generate global and local SHAP plots for the saved Olist model."""

import argparse
from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd

from module_olist.config import FIGURES_DIR, INTERIM_DATA_DIR, MODELS_DIR
from module_olist.modeling.interpret import explain_predictions
from module_olist.modeling.predict import load_model_artifact

DEFAULT_DATASET_PATH = INTERIM_DATA_DIR / "dataset.csv"
DEFAULT_MODEL_PATH = MODELS_DIR / "model.pkl"


def generate_explanations(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    output_dir: Path = FIGURES_DIR,
    sample_size: int = 100,
    sample_position: int = 0,
) -> dict[str, Path]:
    """Save global and individual SHAP plots for a sample of orders."""
    import shap

    if sample_size < 1:
        raise ValueError("sample_size must be at least 1.")

    artifact = load_model_artifact(model_path)
    data = pd.read_csv(dataset_path)
    if data.empty:
        raise ValueError("The dataset to explain is empty.")

    required_features = artifact["features"]
    missing = set(required_features).difference(data.columns)
    if missing:
        raise ValueError(f"The dataset is missing required features: {sorted(missing)}")

    sample = data[required_features].sample(n=min(sample_size, len(data)), random_state=42)
    if not 0 <= sample_position < len(sample):
        raise ValueError("sample_position is outside the sampled rows.")

    explanation = explain_predictions(artifact["model"], sample)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "global_bar": output_dir / "shap_global_bar.png",
        "beeswarm": output_dir / "shap_beeswarm.png",
        "waterfall": output_dir / "shap_waterfall.png",
    }

    shap.plots.bar(explanation, max_display=10, show=False)
    plt.savefig(paths["global_bar"], dpi=300, bbox_inches="tight")
    plt.close()

    shap.plots.beeswarm(explanation, max_display=10, show=False)
    plt.savefig(paths["beeswarm"], dpi=300, bbox_inches="tight")
    plt.close()

    shap.plots.waterfall(explanation[sample_position], max_display=10, show=False)
    plt.savefig(paths["waterfall"], dpi=300, bbox_inches="tight")
    plt.close()

    logger.success("SHAP plots saved to {}", output_dir)
    return paths


def main() -> None:
    """Parse command-line options and generate SHAP plots."""
    parser = argparse.ArgumentParser(description="Explain saved Olist model predictions.")
    parser.add_argument("--input", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--sample-position", type=int, default=0)
    args = parser.parse_args()
    generate_explanations(
        dataset_path=args.input,
        model_path=args.model,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        sample_position=args.sample_position,
    )


if __name__ == "__main__":
    main()
