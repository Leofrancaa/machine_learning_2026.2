"""Escolhas feitas na validação: a largura σ do score e o limiar de decisão.

O conjunto de teste não é usado para escolher nada; ele fica reservado para evaluate.py.
"""

from loguru import logger
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import torch

from module_vae.config import SIGMAS, THRESHOLD_PERCENTILE
from module_vae.features import anomaly_score


def sigmas_for(image_size: int) -> list[float]:
    """Larguras de suavização testadas, escaladas para a resolução da imagem.

    SIGMAS está em pixels de uma imagem de 64 px; em 128 px, cada σ dobra.
    """
    return [s * image_size / 64 for s in SIGMAS]


def select_sigma(
    x_val: torch.Tensor,
    rec_val: torch.Tensor,
    y_val: np.ndarray,
    sigmas: list[float] | None = None,
    mask: torch.Tensor | None = None,
) -> tuple[float, dict[float, float]]:
    """Escolhe a largura da suavização do resíduo pela AUC na validação.

    Args:
        x_val (torch.Tensor): Exames de validação (normais e positivos).
        rec_val (torch.Tensor): Reconstruções desses exames.
        y_val (np.ndarray): Rótulos de validação.
        sigmas (list[float] | None): Larguras testadas, em pixels. Por padrão, SIGMAS
            escalado para a resolução de x_val.
        mask (torch.Tensor | None): Máscara dos pulmões (ablação C1).

    Returns:
        tuple[float, dict[float, float]]: O σ escolhido e a AUC de validação de cada σ.
    """
    sigmas = sigmas or sigmas_for(x_val.shape[-1])
    auc_by_sigma = {
        s: roc_auc_score(y_val, anomaly_score(x_val, rec_val, s, mask)) for s in sigmas
    }
    best_sigma = max(auc_by_sigma, key=auc_by_sigma.get)

    logger.info(f"AUC na validação por σ: { {s: round(a, 3) for s, a in auc_by_sigma.items()} }")
    logger.success(f"σ escolhido: {best_sigma} (AUC val {auc_by_sigma[best_sigma]:.3f})")

    return best_sigma, auc_by_sigma


def calibrate_threshold(
    score_val: np.ndarray,
    y_val: np.ndarray,
    percentile: float = THRESHOLD_PERCENTILE,
) -> float:
    """Limiar sem rótulos de pneumonia: um percentil dos scores dos normais da validação.

    Com o percentil 95, espera-se que ~5% dos exames normais passem do limiar (falsos
    alarmes). É a opção realista para um detector de anomalias, que só conhece normais:
    você escolhe quantos falsos alarmes aceita.
    """
    threshold = float(np.percentile(score_val[y_val == 0], percentile))
    logger.info(f"Limiar (percentil {percentile} dos normais da validação): {threshold:.6f}")

    return threshold


def youden_threshold(score_val: np.ndarray, y_val: np.ndarray) -> float:
    """Limiar com rótulos: maximiza sensibilidade + especificidade − 1 na validação."""
    fpr, tpr, thresholds = roc_curve(y_val, score_val)
    return float(thresholds[np.argmax(tpr - fpr)])
