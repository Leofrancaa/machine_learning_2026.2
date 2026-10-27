"""Arquitetura do VAE e da CNN supervisionada usada como referência.

raio-X x ─► Encoder (5 conv: 128→64→32→16→8→4 px) ─► μ, log σ² ─► z = μ + σ ⊙ ε
         ─► Decoder (5 deconv: 4→8→16→32→64→128 px) ─► reconstrução x̂

Loss = BCE(x, x̂) + β · KL( N(μ, σ²) ‖ N(0, I) )

A rede se adapta à resolução: cada convolução reduz a imagem pela metade até 4 px, então
em 64 px encoder e decoder têm uma camada a menos. O código latente e o "gargalo" de 256
canais × 4 × 4 px são sempre os mesmos.
"""

from itertools import pairwise
import math

from loguru import logger
import torch
from torch import nn
import torch.nn.functional as F

FLAT_SIZE = 256 * 4 * 4  # o encoder sempre termina em 256 canais × 4 × 4 px


def channels_for(image_size: int) -> list[int]:
    """Canais de cada camada do encoder. Cada convolução reduz a imagem pela metade até 4 px.

    64 px → [1, 32, 64, 128, 256]; 128 px → [1, 16, 32, 64, 128, 256].
    """
    n_blocks = int(math.log2(image_size // 4))
    if image_size != 4 * 2**n_blocks:
        raise ValueError(f"A resolução precisa ser uma potência de 2 (recebido: {image_size}).")
    return [1] + [256 // 2 ** (n_blocks - 1 - i) for i in range(n_blocks)]


def conv_block(c_in: int, c_out: int) -> list[nn.Module]:
    """Convolução com passo 2 (reduz a imagem pela metade) + batch norm + LeakyReLU."""
    return [
        nn.Conv2d(c_in, c_out, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(c_out),
        nn.LeakyReLU(0.2),
    ]


class VAE(nn.Module):
    """Autoencoder variacional convolucional para imagens 1×H×H (H = 64, 128, ...).

    Com deterministic=True, a mesma rede vira um autoencoder comum (z = μ, sem KL), usado
    como baseline para isolar o efeito da parte variacional.
    """

    def __init__(self, latent_dim: int = 8, deterministic: bool = False, image_size: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.deterministic = deterministic
        self.image_size = image_size
        channels = channels_for(image_size)

        layers = []
        for c_in, c_out in pairwise(channels):
            layers += conv_block(c_in, c_out)
        self.encoder = nn.Sequential(*layers, nn.Flatten())  # → 256×4×4 = 4096
        self.fc_mu = nn.Linear(FLAT_SIZE, latent_dim)
        self.fc_logvar = nn.Linear(FLAT_SIZE, latent_dim)

        self.fc_decoder = nn.Linear(latent_dim, FLAT_SIZE)
        reversed_channels = channels[::-1]  # 64 px: 256, 128, 64, 32, 1
        layers = []
        for c_in, c_out in pairwise(reversed_channels[:-1]):
            layers += [
                nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(c_out),
                nn.LeakyReLU(0.2),
            ]
        layers += [
            nn.ConvTranspose2d(reversed_channels[-2], 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # pixels em [0, 1]
        ]
        self.decoder = nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # Truque da reparametrização: z = μ + σ·ε. Fora do treino (ou no autoencoder
        # comum), usamos z = μ, o que deixa a inferência determinística.
        if self.deterministic or not self.training:
            return mu
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.fc_decoder(z).view(-1, 256, 4, 4))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_terms(
    x: torch.Tensor,
    x_rec: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calcula os dois termos da loss do VAE, um valor por imagem.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - reconstrução: BCE somada sobre os pixels de cada imagem;
            - KL entre N(μ, σ²) e N(0, I), fórmula fechada, somada sobre as dimensões
              latentes.
    """
    reconstruction = F.binary_cross_entropy(x_rec, x, reduction="none").sum(dim=(1, 2, 3))
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
    return reconstruction, kl


def create_supervised_cnn(image_size: int = 128) -> nn.Sequential:
    """Mesmo encoder do VAE + uma camada de classificação (logit de pneumonia).

    É a referência supervisionada: treinada com normais e pneumonias, dá o "teto"
    esperado para a tarefa.
    """
    layers = []
    for c_in, c_out in pairwise(channels_for(image_size)):
        layers += conv_block(c_in, c_out)
    return nn.Sequential(*layers, nn.Flatten(), nn.Dropout(0.3), nn.Linear(FLAT_SIZE, 1))


def describe_architecture(model: VAE) -> None:
    """Registra no log o formato do tensor depois de cada camada e o total de parâmetros."""
    model = model.cpu().eval()
    rows = []
    with torch.no_grad():
        h = torch.zeros(1, 1, model.image_size, model.image_size)
        rows.append(("entrada (raio-X)", tuple(h.shape[1:])))
        for layer in model.encoder:
            h = layer(h)
            if isinstance(layer, (nn.Conv2d, nn.Flatten)):
                rows.append((f"encoder · {type(layer).__name__}", tuple(h.shape[1:])))
        rows.append(("μ e log σ²", (model.latent_dim,)))
        h = model.fc_decoder(torch.zeros(1, model.latent_dim)).view(-1, 256, 4, 4)
        rows.append(("decoder · Linear + reshape", tuple(h.shape[1:])))
        for layer in model.decoder:
            h = layer(h)
            if isinstance(layer, nn.ConvTranspose2d):
                rows.append(("decoder · ConvTranspose2d", tuple(h.shape[1:])))

    n_params = sum(p.numel() for p in model.parameters())
    table = "\n".join(f"  {name:<30} {shape}" for name, shape in rows)
    logger.info(f"Arquitetura do VAE:\n{table}\n  Total de parâmetros: {n_params:,}")
