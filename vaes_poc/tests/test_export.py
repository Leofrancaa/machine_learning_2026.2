from types import SimpleNamespace

import numpy as np
import pandas as pd

from module_vae.export import ORIGINAL_SIZE, RunExport, boxes_percent, sample_exams


def test_model_info_exports_parameters_from_the_trained_run(tmp_path):
    run = RunExport.__new__(RunExport)
    run.paths = SimpleNamespace(name="future_run", reports=tmp_path)
    run.size = 256
    run.meta = {
        "views": ["PA"],
        "sigma": 3,
        "threshold": 0.1,
        "latent_dim": 4,
        "auc_val": 0.77,
        "version": "future",
        "epochs": 12,
        "threshold_rule": "regra da execução futura",
        "normalize": True,
        "lung_mask": True,
    }
    run.splits = SimpleNamespace(y_test=[0, 1])

    info = run.model_info()

    assert info["image_size"] == 256
    assert info["original_image_size"] == ORIGINAL_SIZE
    assert info["epochs"] == 12
    assert info["version"] == "future"
    assert info["threshold_rule"] == run.meta["threshold_rule"]
    assert info["n_test"] == 2
    assert info["normalize"] is True
    assert info["lung_mask"] is True


def test_sample_exams_keeps_view_share_of_each_class():
    # Normais: 80% PA; opacidades: 75% AP (como o atalho do RSNA)
    y = np.array([0] * 100 + [1] * 40)
    views = np.array(["PA"] * 80 + ["AP"] * 20 + ["AP"] * 30 + ["PA"] * 10)

    idx = sample_exams(y, views, {0: 16, 1: 4}, np.random.default_rng(0))

    assert len(idx) == len(set(idx)) == 20
    picked = pd.crosstab(y[idx], views[idx])
    assert picked.loc[0, "PA"] == 13 and picked.loc[0, "AP"] == 3
    assert picked.loc[1, "AP"] == 3 and picked.loc[1, "PA"] == 1


def test_sample_exams_with_a_single_view():
    y = np.array([0] * 30 + [1] * 10)
    views = np.full(40, "PA")

    idx = sample_exams(y, views, {0: 6, 1: 2}, np.random.default_rng(1))

    assert (y[idx] == 0).sum() == 6 and (y[idx] == 1).sum() == 2


def test_boxes_percent_converts_pixels_to_percent():
    boxes = pd.DataFrame(
        {
            "exam_index": [3, 3, 5],
            "x": [32.0, 0.0, 1.0],
            "y": [64.0, 0.0, 1.0],
            "width": [16.0, 128.0, 1.0],
            "height": [8.0, 128.0, 1.0],
        }
    )

    assert boxes_percent(boxes, 3, 128) == [[25.0, 50.0, 12.5, 6.25], [0.0, 0.0, 100.0, 100.0]]
    assert boxes_percent(boxes, 4, 128) == []
