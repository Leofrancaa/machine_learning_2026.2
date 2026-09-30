"""SHAP explanations for the trained Olist classifier."""

import numpy as np
import pandas as pd
from scipy import sparse


def explain_predictions(pipeline, features: pd.DataFrame):
    """Explain the positive class in the estimator's raw output units."""
    import shap

    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = preprocessor.transform(features)
    if sparse.issparse(transformed):
        transformed = transformed.toarray()

    feature_names = preprocessor.get_feature_names_out().tolist()
    transformed_features = pd.DataFrame(
        transformed,
        columns=feature_names,
        index=features.index,
    )
    explanation = shap.TreeExplainer(pipeline.named_steps["model"])(transformed_features)
    values = np.asarray(explanation.values)
    base_values = np.asarray(explanation.base_values)

    if values.ndim == 3:
        if values.shape[2] != 2:
            raise ValueError("Expected SHAP values for a binary classifier.")
        values = values[:, :, 1]
        if base_values.ndim == 2:
            base_values = base_values[:, 1]
        elif base_values.shape == (2,):
            base_values = base_values[1]
        else:
            raise ValueError("Unexpected SHAP base values for a binary classifier.")

    if values.shape != transformed_features.shape:
        raise ValueError("SHAP values do not match the transformed feature matrix.")

    return shap.Explanation(
        values=values,
        base_values=base_values,
        data=transformed_features.to_numpy(),
        feature_names=feature_names,
    )
