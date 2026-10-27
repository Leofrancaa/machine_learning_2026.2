"""Do resíduo de reconstrução ao score de anomalia.

O resíduo r = x − x̂ mistura duas coisas de natureza diferente:

- baixa frequência (regiões inteiras mais claras ou mais escuras que o esperado): é aqui
  que uma opacidade de pneumonia aparece;
- alta frequência (bordas de costelas, vasos): o VAE perde esse detalhe em todos os
  exames, porque reconstrói borrado. Como os normais têm mais detalhe fino, essa parte do
  erro é maior nos normais e cancela o sinal útil.

Por isso o score suaviza o resíduo com um filtro gaussiano de largura σ antes de medir o
erro. Com σ = 0, o score volta a ser o erro quadrático médio comum.
"""

import numpy as np
import torch
import torch.nn.functional as F


def gaussian_kernel1d(sigma: float) -> torch.Tensor:
    """Pesos de um filtro gaussiano 1D com tamanho ímpar ≈ 4σ, somando 1."""
    kernel_size = int(4 * sigma) | 1
    half = (kernel_size - 1) / 2
    x = torch.linspace(-half, half, kernel_size)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def smooth(img: torch.Tensor, sigma: float, batch_size: int = 128) -> torch.Tensor:
    """Aplica um filtro gaussiano de largura sigma em imagens (N, 1, H, W).

    O filtro gaussiano 2D é separável: um filtro 1D na horizontal seguido de outro na
    vertical dá exatamente o mesmo resultado, com 2·k contas por pixel em vez de k². Em
    128 px, σ chega a 16 px e o kernel a 65 × 65, então isso faz muita diferença. As
    imagens são processadas em lotes para limitar o uso de memória.
    """
    if sigma <= 0 or len(img) == 0:
        return img

    kernel = gaussian_kernel1d(sigma).to(img)
    pad = len(kernel) // 2
    horizontal, vertical = kernel.view(1, 1, 1, -1), kernel.view(1, 1, -1, 1)

    out = []
    for i in range(0, len(img), batch_size):
        x = F.pad(img[i : i + batch_size], (pad, pad, pad, pad), mode="reflect")
        out.append(F.conv2d(F.conv2d(x, horizontal), vertical))
    return torch.cat(out)


def anomaly_map(
    x: torch.Tensor, x_rec: torch.Tensor, sigma: float, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Mapa de calor: resíduo suavizado e elevado ao quadrado, pixel a pixel.

    Args:
        x (torch.Tensor): Exames originais (N, 1, H, W).
        x_rec (torch.Tensor): Reconstruções do modelo, mesmo formato.
        sigma (float): Largura da suavização do resíduo, em pixels.
        mask (torch.Tensor | None): Máscara dos pulmões (H, W), ablação C1. Fora dela, o
            mapa é zero. A suavização vem antes, então o mapa não ganha uma "borda".

    Returns:
        torch.Tensor: Mapa (N, 1, H, W). Valores altos indicam onde o exame foge do
            "normal" reconstruído.
    """
    heatmap = smooth(x - x_rec, sigma) ** 2
    return heatmap if mask is None else heatmap * mask.to(heatmap)


def anomaly_score(
    x: torch.Tensor, x_rec: torch.Tensor, sigma: float, mask: torch.Tensor | None = None
) -> np.ndarray:
    """Score de anomalia de cada exame: a média do mapa de calor.

    Com a máscara, a média continua sobre a imagem inteira; como a área da máscara é a
    mesma para todos, a ordem dos exames é a mesma da média só dentro dela.

    Returns:
        np.ndarray: Score (N,). Quanto maior, mais o exame foge do padrão normal.
    """
    return anomaly_map(x, x_rec, sigma, mask).mean(dim=(1, 2, 3)).numpy()


def detail_level(x: torch.Tensor, sigma: float = 1.5) -> np.ndarray:
    """Nível de detalhe fino de cada imagem: |x − suavizar(x)| médio.

    É a heurística "menos detalhe = pneumonia". Só serve de referência: ela foi escolhida
    olhando os rótulos de pneumonia.
    """
    return (x - smooth(x, sigma)).abs().mean(dim=(1, 2, 3)).numpy()


def brightness(x: torch.Tensor) -> np.ndarray:
    """Brilho médio de cada imagem."""
    return x.mean(dim=(1, 2, 3)).numpy()
