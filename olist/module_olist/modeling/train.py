"""Cross-validated training for delivery-delay classifiers."""

from collections.abc import Mapping
import json
from pathlib import Path
import pickle

import joblib
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from module_olist.config import INTERIM_DATA_DIR, MODELS_DIR
from module_olist.modeling.evaluate import evaluate_model
from module_olist.modeling.pipeline import (
    create_gradient_boosting_pipeline,
    create_lightgbm_pipeline,
    create_xgboost_pipeline,
)
from module_olist.modeling.split import FEATURES, split_data

DATASET_PATH = INTERIM_DATA_DIR / "dataset.csv"
MODEL_PATH = MODELS_DIR / "model.pkl"
SELECTED_MODEL_PATH = MODELS_DIR / "best_model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.01), 2)


def create_models() -> dict[str, object]:
    """Create the candidate model pipelines used during cross-validation."""
    return {
        "xgboost": create_xgboost_pipeline(),
        "lightgbm": create_lightgbm_pipeline(),
        "gradient_boosting": create_gradient_boosting_pipeline(),
    }


def _metrics_at_threshold(y_true, y_proba, threshold: float) -> dict[str, float]:
    """Calculate binary classification metrics at a fixed threshold."""
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
    }


def select_threshold(y_true, y_proba) -> float:
    """Select the probability threshold that maximizes F1."""
    scores = [
        f1_score(
            y_true,
            (np.asarray(y_proba) >= threshold).astype(int),
            zero_division=0,
        )
        for threshold in THRESHOLDS
    ]
    return float(THRESHOLDS[int(np.argmax(scores))])


def cross_validate_models(
    X_train,
    y_train,
    models: Mapping[str, object],
    n_splits: int = 5,
) -> dict[str, dict[str, float]]:
    """Evaluate candidates with stratified out-of-fold predictions.

    Every row is predicted by a model that was not fitted on that row. The
    resulting probabilities are used to select the threshold without consulting
    the held-out test set.
    """
    class_counts = pd.Series(y_train).value_counts()
    if len(class_counts) != 2:
        raise ValueError("Cross-validation requires a binary target with both classes.")
    if class_counts.min() < n_splits:
        raise ValueError(
            f"Each target class must have at least {n_splits} rows for {n_splits}-fold CV."
        )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {}

    for name, model in models.items():
        logger.info("Running {}-fold cross-validation for {}...", n_splits, name)
        oof_proba = cross_val_predict(
            model,
            X_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=None,
        )[:, 1]
        threshold = select_threshold(y_train, oof_proba)
        results[name] = _metrics_at_threshold(y_train, oof_proba, threshold)
        _log_metrics(name, results[name], stage="CV")

    return results


def fit_models(X_train, y_train, models: Mapping[str, object]) -> dict[str, object]:
    """Fit fresh copies of all candidate pipelines on the complete training set."""
    fitted_models = {}
    for name, model in models.items():
        logger.info("Fitting {} on the complete training set...", name)
        fitted_models[name] = clone(model).fit(X_train, y_train)
    return fitted_models


def evaluate_models(
    X_test,
    y_test,
    models: Mapping[str, object],
    cv_results: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    """Evaluate fitted models once on test data using CV-selected thresholds."""
    results = {}
    for name, model in models.items():
        if name not in cv_results:
            raise ValueError(f"Missing cross-validation results for model: {name}")

        threshold = float(cv_results[name]["threshold"])
        results[name] = evaluate_model(model, name, X_test, y_test, threshold)
        _log_metrics(name, results[name], stage="test")

    return results


def _log_metrics(name: str, metrics: Mapping[str, float], stage: str) -> None:
    """Log a model's metrics consistently."""
    logger.info(
        "{} [{}] | threshold={:.2f} accuracy={:.4f} precision={:.4f} "
        "recall={:.4f} f1={:.4f} roc_auc={:.4f} pr_auc={:.4f}",
        name,
        stage,
        metrics["threshold"],
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1_score"],
        metrics["roc_auc"],
        metrics["pr_auc"],
    )


def save_model_artifact(
    model_path: Path,
    model_name: str,
    model: object,
    cv_metrics: Mapping[str, float],
    test_metrics: Mapping[str, float],
) -> None:
    """Persist the selected pipeline, threshold, features and training metrics."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model_name": model_name,
        "model": model,
        "threshold": float(cv_metrics["threshold"]),
        "features": FEATURES,
        "cv_metrics": dict(cv_metrics),
        "test_metrics": dict(test_metrics),
    }
    with model_path.open("wb") as model_file:
        pickle.dump(artifact, model_file)
    logger.success("Model artifact saved to {}", model_path)


def save_selected_model(
    model_path: Path,
    metadata_path: Path,
    model_name: str,
    model: object,
    threshold: float,
) -> None:
    """Save the selected pipeline and inference metadata as separate files."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metadata = {
        "model_name": model_name,
        "threshold": float(threshold),
        "selection_metric": "f1_oof",
        "threshold_metric": "f1",
        "features": FEATURES,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    logger.success("Selected model and metadata saved to {} and {}", model_path, metadata_path)


def main(
    dataset_path: Path = DATASET_PATH,
    model_path: Path = MODEL_PATH,
    n_splits: int = 5,
) -> None:
    """Run cross-validation, final fitting, test evaluation and persistence."""
    logger.info("Loading modeling dataset from {}...", dataset_path)
    data = pd.read_csv(dataset_path)
    X_train, X_test, y_train, y_test = split_data(data)
    test_features_path = dataset_path.parent / "X_test.csv"
    X_test.to_csv(test_features_path, index=False)

    logger.info(
        "Training rows: {:,} | test rows: {:,} | positive rate: {:.2%}",
        len(X_train),
        len(X_test),
        y_train.mean(),
    )
    candidates = create_models()
    cv_results = cross_validate_models(X_train, y_train, candidates, n_splits=n_splits)

    best_name = max(cv_results, key=lambda name: cv_results[name]["f1_score"])
    logger.info("Best candidate selected from CV: {}", best_name)

    fitted_models = fit_models(X_train, y_train, {best_name: candidates[best_name]})
    test_results = evaluate_models(X_test, y_test, fitted_models, cv_results)
    save_model_artifact(
        model_path,
        best_name,
        fitted_models[best_name],
        cv_results[best_name],
        test_results[best_name],
    )
    save_selected_model(
        SELECTED_MODEL_PATH,
        METADATA_PATH,
        best_name,
        fitted_models[best_name],
        cv_results[best_name]["threshold"],
    )
    logger.success("Cross-validated training completed.")


if __name__ == "__main__":
    main()
