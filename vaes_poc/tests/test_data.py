from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from module_vae.config import run_name
from module_vae.dataset import exclusion_reasons
from module_vae.modeling.split import split_rsna

CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "rsna_128.npz"

requires_cache = pytest.mark.skipif(
    not CACHE_PATH.exists(), reason="Cache do RSNA não criado; rode `make data`"
)


@requires_cache
def test_cache_has_one_image_per_patient():
    cache = np.load(CACHE_PATH)

    assert cache["images"].shape == (26684, 128, 128)
    assert cache["images"].dtype == np.uint8
    assert len(np.unique(cache["patient_ids"])) == 26684


@requires_cache
def test_cache_labels_and_views():
    cache = np.load(CACHE_PATH)

    assert set(cache["labels"]) == {"normal", "opacity", "other"}
    assert set(cache["views"]) == {"AP", "PA"}


def make_fake_rsna(n_per_group: int = 20):
    # Cada imagem é preenchida com o próprio índice, para saber de onde ela veio.
    groups = [(label, view) for label in ["normal", "opacity", "other"] for view in ["AP", "PA"]]
    meta = pd.DataFrame(
        [
            {
                "patientId": f"p{i}_{j}",
                "label": label,
                "view": view,
                "pixel_spacing": 0.143 if j % 2 else 0.168,
                "exclusion": "",
            }
            for i, (label, view) in enumerate(groups)
            for j in range(n_per_group)
        ]
    )
    images = np.repeat(np.arange(len(meta), dtype=np.uint8)[:, None, None], 4, axis=1)
    images = np.repeat(images, 4, axis=2)
    opacity_ids = meta.loc[meta["label"] == "opacity", "patientId"]
    boxes = pd.DataFrame(
        {"patientId": opacity_ids, "x": 1.0, "y": 1.0, "width": 2.0, "height": 2.0}
    )
    return images, meta, boxes


def test_split_trains_vae_only_on_normals():
    images, meta, boxes = make_fake_rsna()

    splits = split_rsna(images, meta, boxes, views=["AP", "PA"])

    origin = splits.x_train_normal[:, 0, 0, 0].mul(255).round().int().numpy()
    assert (meta["label"].to_numpy()[origin] == "normal").all()
    assert set(np.unique(splits.y_test)) == {0, 1}
    assert len(splits.x_test_other) > 0


def test_pa_filter_keeps_only_pa_and_is_subset_of_full_test():
    images, meta, boxes = make_fake_rsna()

    pa = split_rsna(images, meta, boxes, views=["PA"])
    full = split_rsna(images, meta, boxes, views=["AP", "PA"])

    assert set(pa.views_test) == {"PA"}
    pa_test = set(pa.x_test[:, 0, 0, 0].tolist())
    full_test = set(full.x_test[:, 0, 0, 0].tolist())
    assert pa_test <= full_test


def test_boxes_point_to_opacity_exams():
    images, meta, boxes = make_fake_rsna()

    splits = split_rsna(images, meta, boxes, views=["AP", "PA"])

    assert (splits.y_test[splits.boxes_test["exam_index"]] == 1).all()


def test_validation_boxes_point_to_validation_opacities():
    images, meta, boxes = make_fake_rsna()

    splits = split_rsna(images, meta, boxes, views=["AP", "PA"])

    assert len(splits.boxes_val) > 0
    assert (splits.y_val[splits.boxes_val["exam_index"]] == 1).all()
    assert len(splits.equipment_val) == len(splits.y_val)
    assert set(splits.equipment_test) <= {"0.143", "0.168"}


def test_hygiene_removes_out_of_scope_exams_and_keeps_the_split():
    images, meta, boxes = make_fake_rsna()
    meta.loc[meta.index % 5 == 0, "exclusion"] = "menor de 18"
    excluded_ids = set(np.where(meta["exclusion"] != "")[0])

    clean = split_rsna(images, meta, boxes, views=["AP", "PA"])
    v1 = split_rsna(images, meta, boxes, views=["AP", "PA"], hygiene=False)

    # O teste com higiene é o teste da v1 sem os exames fora do escopo.
    clean_test = set(clean.x_test[:, 0, 0, 0].mul(255).round().int().tolist())
    v1_test = set(v1.x_test[:, 0, 0, 0].mul(255).round().int().tolist())
    assert clean_test == v1_test - excluded_ids
    assert not clean_test & excluded_ids
    assert clean.excluded.to_numpy().sum() > 0
    assert v1.excluded.to_numpy().sum() == 0


def test_exclusion_reasons():
    images = np.zeros((5, 4, 4), dtype=np.uint8)
    images[1] = images[2] = 7  # p1 e p2: o mesmo exame com dois ids
    images[3], images[4] = 1, 2
    meta = pd.DataFrame(
        {"patientId": ["p0", "p2", "p1", "p3", "p4"], "age": [30, 40, 40, 150, 10]}
    )

    reasons = exclusion_reasons(images, meta)

    # fica o de menor patientId (p1); idade 150 é impossível; 10 anos está fora do escopo
    assert reasons.tolist() == ["", "duplicata", "", "idade ausente", "menor de 18"]


def test_run_name_keeps_v1_folders_and_adds_version_suffix():
    assert run_name(["PA"]) == "rsna_pa"
    assert run_name(["PA"], "v1") == "rsna_pa"
    assert run_name(["PA", "AP"], "v2") == "rsna_ap_pa_v2"
