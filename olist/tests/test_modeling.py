import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from module_olist.modeling.train import (
    cross_validate_models,
    evaluate_models,
    fit_models,
)


def test_cross_validation_selects_threshold_without_fitting_input_model():
    X, y = make_classification(
        n_samples=80,
        n_features=5,
        weights=[0.75, 0.25],
        random_state=42,
    )
    candidate = LogisticRegression(max_iter=500, random_state=42)

    cv_results = cross_validate_models(X, y, {"logistic": candidate}, n_splits=4)

    assert "coef_" not in vars(candidate)
    assert 0.05 <= cv_results["logistic"]["threshold"] <= 0.95
    assert 0.0 <= cv_results["logistic"]["roc_auc"] <= 1.0


def test_test_evaluation_reuses_cross_validation_threshold():
    X, y = make_classification(
        n_samples=80,
        n_features=5,
        weights=[0.75, 0.25],
        random_state=42,
    )
    candidates = {"logistic": LogisticRegression(max_iter=500, random_state=42)}
    fitted_models = fit_models(X[:60], y[:60], candidates)
    cv_results = {"logistic": {"threshold": 0.73}}

    test_results = evaluate_models(X[60:], y[60:], fitted_models, cv_results)

    assert np.isclose(test_results["logistic"]["threshold"], 0.73)
