"""Inferência com um VAE treinado: reconstrução, score, limiar e posição na fila."""

import json
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import torch

from module_vae.config import DEVICE
from module_vae.features import anomaly_map
from module_vae.modeling.architecture import VAE


@torch.no_grad()
def reconstruct(
    model: VAE, x: torch.Tensor, batch_size: int = 256, device: torch.device = DEVICE
) -> torch.Tensor:
    """Reconstrói as imagens em lotes e devolve o resultado na CPU."""
    model.eval()
    return torch.cat(
        [model(x[i : i + batch_size].to(device))[0].cpu() for i in range(0, len(x), batch_size)]
    )


def load_model(
    model_path: Path, metadata_path: Path, device: torch.device = DEVICE
) -> tuple[VAE, dict]:
    """Carrega o VAE treinado e os metadados da inferência (sigma, limiar...)."""
    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    model = VAE(metadata["latent_dim"], image_size=metadata["image_size"])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model = model.to(device).eval()

    # Máscara dos pulmões (ablação C1), salva ao lado do modelo
    mask_path = model_path.parent / "lung_mask.npy"
    if metadata.get("lung_mask") and mask_path.exists():
        metadata["mask"] = torch.from_numpy(np.load(mask_path))

    logger.info(f"Modelo carregado: {model_path}")
    logger.info(f"σ: {metadata['sigma']} | limiar: {metadata['threshold']:.6f}")

    return model, metadata


def predict(
    model: VAE,
    x: torch.Tensor,
    sigma: float,
    threshold: float,
    mask: torch.Tensor | None = None,
) -> tuple[pd.DataFrame, torch.Tensor, torch.Tensor]:
    """Calcula score, decisão e posição na fila de triagem de cada exame.

    Args:
        model (VAE): Modelo treinado.
        x (torch.Tensor): Exames (N, 1, H, H) com pixels em [0, 1].
        sigma (float): Largura da suavização do resíduo escolhida na validação.
        threshold (float): Limiar de decisão calibrado na validação.
        mask (torch.Tensor | None): Máscara dos pulmões (ablação C1).

    Returns:
        tuple[pd.DataFrame, torch.Tensor, torch.Tensor]:
            - predictions: anomaly_score, prediction (1 = revisar com prioridade) e
              queue_position (1 = primeiro da fila);
            - x_rec: reconstruções;
            - maps: mapas de calor (o score é a média de cada mapa).
    """
    x_rec = reconstruct(model, x)
    maps = anomaly_map(x, x_rec, sigma, mask)
    score = maps.mean(dim=(1, 2, 3)).numpy()

    predictions = pd.DataFrame(
        {
            "anomaly_score": score,
            "prediction": (score >= threshold).astype(int),
            # Ordenar a fila pelo score: o exame mais "anômalo" é o primeiro a ser visto.
            "queue_position": np.argsort(np.argsort(-score)) + 1,
        }
    )

    return predictions, x_rec, maps
