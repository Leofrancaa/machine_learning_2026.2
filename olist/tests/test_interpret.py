import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
from scipy import sparse

from module_olist.modeling.interpret import explain_predictions


def test_explain_predictions_uses_preprocessed_positive_class(monkeypatch):
    class Preprocessor:
        def transform(self, features):
            assert features.index.tolist() == [4, 9]
            return sparse.csr_matrix([[1.0, 0.0], [0.0, 1.0]])

        def get_feature_names_out(self):
            return np.array(["numeric__price", "categorical__state_SP"])

    class TreeExplainer:
        def __init__(self, model):
            assert model is estimator

        def __call__(self, transformed):
            assert transformed.columns.tolist() == [
                "numeric__price",
                "categorical__state_SP",
            ]
            return SimpleNamespace(
                values=np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
                base_values=np.array([[0.1, 0.9], [0.2, 0.8]]),
            )

    class Explanation:
        def __init__(self, **values):
            self.__dict__.update(values)

    shap = ModuleType("shap")
    shap.TreeExplainer = TreeExplainer
    shap.Explanation = Explanation
    monkeypatch.setitem(sys.modules, "shap", shap)

    estimator = object()
    pipeline = SimpleNamespace(
        named_steps={"preprocessor": Preprocessor(), "model": estimator}
    )
    features = pd.DataFrame({"price": [10.0, 20.0]}, index=[4, 9])

    explanation = explain_predictions(pipeline, features)

    np.testing.assert_array_equal(explanation.values, [[2.0, 4.0], [6.0, 8.0]])
    np.testing.assert_array_equal(explanation.base_values, [0.9, 0.8])
    np.testing.assert_array_equal(explanation.data, [[1.0, 0.0], [0.0, 1.0]])
    assert explanation.feature_names == ["numeric__price", "categorical__state_SP"]
