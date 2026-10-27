import pandas as pd
import pytest
import torch
import torchvision.transforms.functional as TF

from module_vae.features import anomaly_score, smooth
from module_vae.modeling.architecture import VAE, channels_for, loss_terms
from module_vae.modeling.evaluate import evaluate_localization


def test_vae_output_shapes():
    model = VAE(latent_dim=8, image_size=128)
    x = torch.rand(3, 1, 128, 128)

    x_rec, mu, logvar = model(x)

    assert x_rec.shape == x.shape
    assert mu.shape == (3, 8)
    assert logvar.shape == (3, 8)
    assert x_rec.min() >= 0 and x_rec.max() <= 1


def test_64px_network_has_four_blocks():
    assert channels_for(64) == [1, 32, 64, 128, 256]
    assert sum(p.numel() for p in VAE(latent_dim=8, image_size=64).parameters()) == 1_481_809


def test_vae_is_deterministic_in_eval_mode():
    model = VAE(latent_dim=8, image_size=128).eval()
    x = torch.rand(2, 1, 128, 128)

    with torch.no_grad():
        assert torch.equal(model(x)[0], model(x)[0])


def test_kl_is_zero_for_standard_normal():
    x = torch.full((2, 1, 64, 64), 0.5)
    mu, logvar = torch.zeros(2, 8), torch.zeros(2, 8)

    reconstruction, kl = loss_terms(x, x, mu, logvar)

    assert torch.allclose(kl, torch.zeros(2))
    assert (reconstruction > 0).all()


def test_score_without_smoothing_is_mse():
    x, x_rec = torch.rand(4, 1, 64, 64), torch.rand(4, 1, 64, 64)

    expected = ((x - x_rec) ** 2).mean(dim=(1, 2, 3)).numpy()

    assert torch.allclose(torch.tensor(anomaly_score(x, x_rec, sigma=0)), torch.tensor(expected))


def test_smoothing_keeps_constant_residual():
    # Uma diferença uniforme (região inteira mais clara) é exatamente o sinal de baixa
    # frequência que a suavização deve preservar.
    residual = torch.full((1, 1, 64, 64), 0.2)

    assert smooth(residual, sigma=3).shape == residual.shape
    assert torch.allclose(smooth(residual, sigma=3), residual)


def test_localization_of_a_map_concentrated_in_the_box():
    maps = torch.zeros(1, 1, 16, 16)
    maps[0, 0, 4:8, 4:8] = 1.0
    boxes = pd.DataFrame(
        {"exam_index": [0], "x": [4.0], "y": [4.0], "width": [4.0], "height": [4.0]}
    )

    results = evaluate_localization(maps, boxes)

    assert results["pointing_game"] == 1.0
    assert results["energy_in_boxes"] == 1.0
    assert results["box_area_fraction"] == 16 / 256


@pytest.mark.parametrize("sigma", [1, 3, 8, 16])
def test_separable_smoothing_matches_torchvision_blur(sigma):
    # O filtro separável precisa dar o mesmo resultado do gaussian_blur 2D do torchvision.
    img = torch.rand(5, 1, 128, 128)

    expected = TF.gaussian_blur(img, kernel_size=int(4 * sigma) | 1, sigma=sigma)

    assert torch.allclose(smooth(img, sigma, batch_size=2), expected, atol=1e-5)
