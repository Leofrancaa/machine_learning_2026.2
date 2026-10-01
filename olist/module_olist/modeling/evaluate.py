"""Held-out evaluation for the selected Olist classifier."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(model, model_name, X_test, y_test, threshold: float) -> dict[str, float]:
    """Evaluate the final model with the threshold chosen on training folds."""
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    results = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities))
        if len(np.unique(y_test)) == 2 else float("nan"),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
    }
    return results
