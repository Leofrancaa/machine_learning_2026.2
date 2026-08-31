"""Training evaluation utilities for delivery-delay classifiers."""

from loguru import logger
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def evaluate_models(X_test, y_test, models):
    """Evaluate trained binary classifiers and select an F1-optimal threshold.

    Args:
        X_test: Test features accepted by each fitted model.
        y_test: True binary labels for the test set.
        models: Mapping from model name to fitted classifier. Each classifier must
            provide ``predict_proba``.

    Returns:
        A dictionary containing the selected threshold and evaluation metrics for
        each model.
    """
    results = {}

    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]

        best_threshold = None
        best_f1 = -1.0
        best_precision = None
        best_recall = None

        for threshold in np.round(np.arange(0.05, 0.50, 0.01), 2):
            y_pred = (y_proba >= threshold).astype(int)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_precision = precision
                best_recall = recall

        roc_auc = roc_auc_score(y_test, y_proba)
        results[name] = {
            "best_threshold": float(best_threshold),
            "precision": float(best_precision),
            "recall": float(best_recall),
            "f1_score": float(best_f1),
            "roc_auc": float(roc_auc),
        }

        logger.info("Model: {}", name)
        logger.info("Best Threshold: {:.2f}", best_threshold)
        logger.info("Precision: {:.4f}", best_precision)
        logger.info("Recall: {:.4f}", best_recall)
        logger.info("F1 Score: {:.4f}", best_f1)
        logger.info("ROC AUC: {:.4f}", roc_auc)

    return results
