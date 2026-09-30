import sys
from types import ModuleType

import matplotlib.pyplot as plt
import pandas as pd

from module_olist import explain


def test_generate_explanations_saves_global_and_local_plots(monkeypatch, tmp_path):
    input_path = tmp_path / "dataset.csv"
    pd.DataFrame(
        {"order_id": ["o1", "o2"], "price": [10.0, 20.0], "is_late": [0, 1]}
    ).to_csv(input_path, index=False)

    class Explanation:
        def __getitem__(self, position):
            assert position == 0
            return "one order"

    explanation = Explanation()
    sample_columns = []
    monkeypatch.setattr(
        explain,
        "load_model_artifact",
        lambda path: {"model": object(), "features": ["price"]},
    )

    def explain_sample(model, sample):
        sample_columns.extend(sample.columns)
        return explanation

    monkeypatch.setattr(explain, "explain_predictions", explain_sample)

    def draw_plot(*args, **kwargs):
        plt.figure()
        plt.plot([0, 1], [0, 1])

    shap = ModuleType("shap")
    shap.plots = ModuleType("shap.plots")
    shap.plots.bar = draw_plot
    shap.plots.beeswarm = draw_plot
    shap.plots.waterfall = draw_plot
    monkeypatch.setitem(sys.modules, "shap", shap)

    paths = explain.generate_explanations(
        dataset_path=input_path,
        model_path=tmp_path / "model.pkl",
        output_dir=tmp_path / "figures",
    )

    assert sample_columns == ["price"]
    assert set(paths) == {"global_bar", "beeswarm", "waterfall"}
    assert all(path.is_file() for path in paths.values())
