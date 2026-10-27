"""Transformações testadas nas ablações da v2 (docs/08-ablacoes.md), uma de cada vez.

- A, recorte das bordas: tira as molduras pretas (imagem menor dentro de um fundo preto)
  e volta ao tamanho original. 2,6% dos normais PA têm moldura, e ela vira falso alarme.
- B, normalização de intensidade: estica o brilho de cada imagem entre os percentis 1 e
  99 dela. Equipamento e brilho andam juntos no RSNA; a normalização tenta tirar essa
  diferença. É linear (não muda a ordem dos tons), para reduzir o risco de apagar
  opacidades difusas.
- C1, máscara pulmonar média: o score e o mapa de calor só contam dentro dos pulmões,
  definidos uma vez a partir da imagem média dos normais de treino. Sem ela, o mapa
  acende em ombros, cantos e marcadores de texto.

A e B mexem em cada imagem sozinha, sem nada aprendido dos dados, então não há
vazamento entre conjuntos. A máscara usa só os normais do treino.
"""

from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image
from scipy import ndimage

# Uma linha (ou coluna) de moldura é preta e uniforme. O ar fora do corpo também é
# escuro, mas tem ruído; o desvio-padrão baixo separa um do outro.
BORDER_MAX_MEAN = 10  # em tons de 0 a 255
BORDER_MAX_STD = 5
BORDER_MAX_FRACTION = 0.25  # nunca corta mais de 1/4 da imagem de cada lado

INTENSITY_PERCENTILES = (1.0, 99.0)

# Quanto a máscara cresce além do "miolo" escuro dos pulmões, em pixels de uma imagem de
# 64 px (10 px em 128 px): a escala das lesões (largura típica de 27 px em 128 px) e do
# σ do score. Escolhido antes de qualquer resultado das ablações.
LUNG_MASK_DILATION = 5


@dataclass(frozen=True)
class Preprocessing:
    """Quais ablações estão ligadas. Tudo desligado reproduz a v2."""

    crop_borders: bool = False  # A
    normalize: bool = False  # B
    lung_mask: bool = False  # C1 (aplicada no score, não nas imagens)

    def apply(self, images: np.ndarray) -> np.ndarray:
        """Aplica A e B às imagens (N, H, W) em uint8, nessa ordem."""
        if self.crop_borders:
            images = crop_borders(images)
        if self.normalize:
            images = normalize_intensity(images)
        return images

    def as_dict(self) -> dict[str, bool]:
        return asdict(self)

    @classmethod
    def from_metadata(cls, metadata: dict) -> "Preprocessing":
        """As ablações ligadas no treino de um modelo salvo (antes delas: nenhuma)."""
        return cls(**{key: bool(metadata.get(key, False)) for key in asdict(cls())})


def black_margin(lines: np.ndarray, limit: int) -> int:
    """Quantas linhas seguidas, a partir do início, são moldura preta (até limit)."""
    n = 0
    while n < limit and lines[n].mean() <= BORDER_MAX_MEAN and lines[n].std() <= BORDER_MAX_STD:
        n += 1
    return n


def crop_borders(images: np.ndarray) -> np.ndarray:
    """Ablação A: tira a moldura preta de cada imagem e volta ao tamanho original.

    Args:
        images (np.ndarray): Imagens (N, H, W) em uint8.

    Returns:
        np.ndarray: Imagens do mesmo formato. As que não têm moldura ficam iguais.
    """
    size = images.shape[-1]
    limit = int(BORDER_MAX_FRACTION * size)
    out = images.copy()
    for i, img in enumerate(images):
        top, bottom = black_margin(img, limit), black_margin(img[::-1], limit)
        left, right = black_margin(img.T, limit), black_margin(img.T[::-1], limit)
        if top + bottom + left + right == 0:
            continue
        crop = img[top : size - bottom, left : size - right]
        out[i] = np.asarray(Image.fromarray(crop).resize((size, size), Image.Resampling.BILINEAR))
    return out


def normalize_intensity(images: np.ndarray) -> np.ndarray:
    """Ablação B: estica o brilho de cada imagem entre os percentis 1 e 99 dela.

    Args:
        images (np.ndarray): Imagens (N, H, W) em uint8.

    Returns:
        np.ndarray: Imagens em uint8, cada uma ocupando a faixa de 0 a 255.
    """
    low, high = np.percentile(images, INTENSITY_PERCENTILES, axis=(1, 2), keepdims=True)
    scaled = (images.astype(np.float32) - low) / np.maximum(high - low, 1)
    return np.round(255 * np.clip(scaled, 0, 1)).astype(np.uint8)


def dark_cores(mean_image: np.ndarray, threshold: float) -> list[np.ndarray]:
    """Regiões mais escuras que threshold que não tocam a borda da imagem."""
    labels, n = ndimage.label(mean_image < threshold)
    edge = np.r_[labels[0], labels[-1], labels[:, 0], labels[:, -1]]
    touches_edge = set(np.unique(edge))
    return [labels == i for i in range(1, n + 1) if i not in touches_edge]


def lung_mask(mean_image: np.ndarray) -> np.ndarray:
    """Ablação C1: máscara dos dois pulmões a partir da imagem média dos normais de treino.

    Na imagem média, os pulmões são as duas regiões escuras no meio do tórax (o ar dos
    cantos também é escuro, mas toca a borda). O limiar desce até os dois pulmões
    aparecerem separados da borda; esse "miolo" cobre só a parte mais escura, então a
    máscara cresce LUNG_MASK_DILATION pixels para incluir os campos inferiores e as
    bordas dos pulmões.

    Args:
        mean_image (np.ndarray): Imagem média (H, W), com pixels em [0, 1].

    Returns:
        np.ndarray: Máscara booleana (H, W).
    """
    for threshold in np.arange(0.6, 0.1, -0.005):
        cores = dark_cores(mean_image, threshold)
        if len(cores) >= 2:
            break
    else:
        raise ValueError("Não foi possível achar os dois pulmões na imagem média.")

    core = np.logical_or.reduce(sorted(cores, key=lambda c: c.sum())[-2:])
    radius = round(LUNG_MASK_DILATION * mean_image.shape[-1] / 64)
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    mask = ndimage.binary_dilation(core, structure=xx**2 + yy**2 <= radius**2)
    return ndimage.binary_fill_holes(mask)
