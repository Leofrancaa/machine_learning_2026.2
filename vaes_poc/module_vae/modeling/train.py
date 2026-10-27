"""Treino do VAE (e do autoencoder comum) e da CNN supervisionada de referência."""

import json
from pathlib import Path
import time

from loguru import logger
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from module_vae.config import (
    BATCH_SIZE,
    BETA,
    DEVICE,
    EPOCHS,
    KL_WARMUP_EPOCHS,
    LATENT_DIM,
    LEARNING_RATE,
)
from module_vae.modeling.architecture import VAE, create_supervised_cnn, loss_terms


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_autoencoder(
    x_train: torch.Tensor,
    x_val: torch.Tensor,
    seed: int,
    latent_dim: int = LATENT_DIM,
    beta: float = BETA,
    deterministic: bool = False,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    kl_warmup_epochs: int = KL_WARMUP_EPOCHS,
    device: torch.device = DEVICE,
    verbose: bool = True,
) -> tuple[VAE, dict[str, list[float]]]:
    """Treina um VAE (ou, com deterministic=True, um autoencoder comum).

    Args:
        x_train (torch.Tensor): Exames normais de treino (N, 1, H, H).
        x_val (torch.Tensor): Exames normais de validação, só para acompanhar o treino.
        seed (int): Semente da inicialização e da ordem dos lotes.
        latent_dim (int): Tamanho do código latente z.
        beta (float): Peso final do termo KL.
        deterministic (bool): True treina o autoencoder comum (sem KL, z = μ).
        epochs (int): Número de épocas.
        batch_size (int): Tamanho do lote.
        learning_rate (float): Taxa de aprendizado do Adam.
        kl_warmup_epochs (int): Épocas em que o peso do KL sobe de 0 até beta.
        device (torch.device): Onde treinar.
        verbose (bool): Registra o progresso a cada 10 épocas.

    Returns:
        tuple[VAE, dict[str, list[float]]]: O modelo treinado (em modo eval) e o
            histórico de reconstrução e KL por época, no treino e na validação.
    """
    set_seed(seed)
    model = VAE(latent_dim, deterministic, image_size=x_train.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    x_tr, x_vl = x_train.to(device), x_val.to(device)
    history = {"rec_train": [], "kl_train": [], "rec_val": [], "kl_val": []}
    start = time.time()

    for epoch in range(epochs):
        model.train()

        # KL warm-up: o decoder aprende a reconstruir antes de o espaço latente ser
        # "apertado", o que reduz o risco de colapso do posterior.
        kl_weight = 0.0 if deterministic else beta * min(1.0, (epoch + 1) / kl_warmup_epochs)

        perm = torch.randperm(len(x_tr)).to(device)
        sum_rec = sum_kl = 0.0
        for i in range(0, len(perm), batch_size):
            x = x_tr[perm[i : i + batch_size]]
            x_rec, mu, logvar = model(x)
            rec, kl = loss_terms(x, x_rec, mu, logvar)
            loss = (rec + kl_weight * kl).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_rec += rec.sum().item()
            sum_kl += kl.sum().item()
        history["rec_train"].append(sum_rec / len(x_tr))
        history["kl_train"].append(sum_kl / len(x_tr))

        # Validação em lotes: em 128 px, passar todos os exames de uma vez pesa na memória.
        model.eval()
        rec_val, kl_val = [], []
        with torch.no_grad():
            for i in range(0, len(x_vl), 256):
                x = x_vl[i : i + 256]
                x_rec, mu, logvar = model(x)
                rec, kl = loss_terms(x, x_rec, mu, logvar)
                rec_val.append(rec)
                kl_val.append(kl)
        history["rec_val"].append(torch.cat(rec_val).mean().item())
        history["kl_val"].append(torch.cat(kl_val).mean().item())

        if verbose and (epoch + 1) % 10 == 0:
            logger.info(
                f"época {epoch + 1:3d} | reconstrução {history['rec_train'][-1]:7.1f} | "
                f"KL {history['kl_train'][-1]:5.1f} | "
                f"reconstrução (val) {history['rec_val'][-1]:7.1f}"
            )

    model.eval()
    if verbose:
        logger.info(f"Tempo de treino: {time.time() - start:.0f}s")

    return model, history


def train_supervised_cnn(
    x_train: torch.Tensor,
    y_train: np.ndarray,
    seed: int = 0,
    epochs: int = 8,
    device: torch.device = DEVICE,
) -> nn.Module:
    """Treina a CNN supervisionada com normais e pneumonias (referência de "teto")."""
    set_seed(seed)
    model = create_supervised_cnn(image_size=x_train.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Compensa o desbalanceamento: cada positivo pesa (nº de normais / nº de positivos)
    pos_weight = torch.tensor(
        [(y_train == 0).sum() / (y_train == 1).sum()], dtype=torch.float32, device=device
    )
    x_tr = x_train.to(device)
    y_tr = torch.tensor(y_train, dtype=torch.float32, device=device)

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(x_tr)).to(device)
        for i in range(0, len(perm), 64):
            idx = perm[i : i + 64]
            loss = F.binary_cross_entropy_with_logits(
                model(x_tr[idx]).squeeze(1), y_tr[idx], pos_weight=pos_weight
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def save_model(
    model: VAE,
    metadata: dict,
    model_path: Path,
    metadata_path: Path,
) -> None:
    """Salva os pesos do VAE e os metadados necessários para reproduzir o score.

    Args:
        model (VAE): Modelo treinado.
        metadata (dict): Configuração e calibração (latent_dim, sigma, threshold...).
        model_path (Path): Arquivo .pt com o state_dict.
        metadata_path (Path): Arquivo .json com os metadados.
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, model_path)
    logger.success(f"Modelo salvo em: {model_path}")

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)
    logger.success(f"Metadados salvos em: {metadata_path}")
