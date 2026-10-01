"""Model selection and threshold reporting for Olist classifiers."""

from collections.abc import Mapping

import pandas as pd

from module_olist.modeling.train import create_models, cross_validate_models


def summarize_cv(results: Mapping[str, Mapping[str, float]]) -> pd.DataFrame:
    """Summarize each candidate's out-of-fold metrics and selected threshold."""
    return pd.DataFrame.from_dict(results, orient="index").rename_axis("model")


def select_model(X_train, y_train, n_splits: int = 5) -> tuple[str, float, pd.DataFrame]:
    """Select the candidate with the highest out-of-fold F1 score."""
    results = cross_validate_models(X_train, y_train, create_models(), n_splits=n_splits)
    best_name = max(results, key=lambda name: results[name]["f1_score"])
    return best_name, float(results[best_name]["threshold"]), summarize_cv(results)
