import numpy as np
import pandas as pd
import torch

from module_vae.features import anomaly_map
from module_vae.modeling.evaluate import evaluate_localization
from module_vae.preprocessing import (
    Preprocessing,
    crop_borders,
    lung_mask,
    normalize_intensity,
)


def noisy_images(n: int = 2, size: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(60, 200, size=(n, size, size)).astype(np.uint8)


def test_crop_removes_black_frame_and_keeps_size():
    images = noisy_images()
    framed = images.copy()
    framed[0, :8], framed[0, -4:], framed[0, :, :6] = 0, 0, 0  # moldura só no primeiro

    out = crop_borders(framed)

    assert out.shape == framed.shape
    assert out[0].mean(axis=1)[:2].min() > 10  # as primeiras linhas não são mais pretas
    assert np.array_equal(out[1], framed[1])  # sem moldura, nada muda


def test_crop_keeps_dark_but_noisy_air():
    # O ar fora do corpo é escuro, mas tem ruído: não é moldura.
    images = noisy_images(1)
    images[0, :8] = np.random.default_rng(1).integers(0, 25, size=(8, 64))

    assert np.array_equal(crop_borders(images), images)


def test_normalize_stretches_each_image_and_keeps_order():
    images = noisy_images()
    images[1] = images[1] // 4 + 100  # imagem de pouco contraste

    out = normalize_intensity(images)

    for img in out:
        assert np.percentile(img, 1) <= 1 and np.percentile(img, 99) >= 254
    # transformação linear e crescente: a ordem dos tons não muda
    flat_in, flat_out = images[1].ravel().astype(int), out[1].ravel().astype(int)
    order = np.argsort(flat_in, kind="stable")
    assert (np.diff(flat_out[order]) >= 0).all()


def test_preprocessing_off_changes_nothing():
    images = noisy_images()

    assert np.array_equal(Preprocessing().apply(images), images)


def synthetic_mean_chest(size: int = 64) -> np.ndarray:
    # Fundo claro, cantos escuros (ar) e dois "pulmões" escuros no meio.
    yy, xx = np.mgrid[:size, :size] / size
    img = np.full((size, size), 0.7)
    img[(yy < 0.1) | (yy > 0.9)] = 0.15
    for cx in (0.3, 0.7):
        img[((xx - cx) / 0.12) ** 2 + ((yy - 0.45) / 0.25) ** 2 < 1] = 0.3
    return img


def test_lung_mask_finds_both_lungs_and_ignores_dark_edges():
    mean_image = synthetic_mean_chest()

    mask = lung_mask(mean_image)

    assert mask[29, 19] and mask[29, 45]  # centro de cada pulmão
    assert not mask[:3].any() and not mask[-3:].any()  # ar das bordas fica de fora
    assert 0.1 < mask.mean() < 0.6


def test_masked_map_is_zero_outside_the_mask():
    x, x_rec = torch.rand(2, 1, 64, 64), torch.rand(2, 1, 64, 64)
    mask = torch.zeros(64, 64, dtype=torch.bool)
    mask[16:48, 16:48] = True

    maps = anomaly_map(x, x_rec, sigma=2, mask=mask)

    assert (maps[:, :, ~mask] == 0).all()
    assert (maps[:, :, mask] > 0).any()


def test_localization_reports_chance_inside_the_mask():
    maps = torch.rand(1, 1, 16, 16)
    boxes = pd.DataFrame(
        {"exam_index": [0], "x": [0.0], "y": [0.0], "width": [4.0], "height": [4.0]}
    )
    mask = torch.zeros(16, 16, dtype=torch.bool)
    mask[:8, :8] = True

    results = evaluate_localization(maps, boxes, lung_mask=mask)

    assert results["box_area_fraction"] == 16 / 256
    assert results["box_fraction_in_mask"] == 16 / 64


def test_preprocessing_from_metadata():
    v3 = Preprocessing.from_metadata({"normalize": True, "lung_mask": True, "sigma": 4.0})

    assert v3 == Preprocessing(normalize=True, lung_mask=True)
    assert Preprocessing.from_metadata({"sigma": 10.0}) == Preprocessing()  # v1 e v2
