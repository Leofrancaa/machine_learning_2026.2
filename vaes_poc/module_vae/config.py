from dataclasses import dataclass
import os
from pathlib import Path

from loguru import logger
import torch

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RSNA_DIR = RAW_DATA_DIR / "rsna"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"

# Rastreamento dos experimentos no MLflow (local e fora do git): parâmetros e métricas em
# mlflow.db, artefatos (figuras, tabelas, pesos) em mlartifacts/. As variáveis de ambiente
# permitem apontar para outro lugar, por exemplo num teste rápido que não deve aparecer
# junto das execuções de verdade.
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{PROJ_ROOT / 'mlflow.db'}")
MLFLOW_ARTIFACTS_DIR = Path(os.environ.get("VAE_MLFLOW_ARTIFACTS", PROJ_ROOT / "mlartifacts"))

# Versão dos resultados. Cada versão grava em pastas próprias (models/rsna_pa_v2,
# reports/rsna_pa_v2...); a v1 ocupa as pastas sem sufixo (models/rsna_pa, reports/rsna_pa).
VERSION = "v2"

# Resolução das imagens. Os DICOMs do RSNA têm 1024 px; em 64 px, opacidades pequenas
# somem.
IMAGE_SIZE = 128

# Higiene dos dados (a partir da v2, ver docs/07-higiene-v2.md)
MIN_AGE = 18  # escopo da PoC: adultos
MAX_VALID_AGE = 100  # idades acima disso (148 a 155 anos no RSNA) são tratadas como ausentes

# Regra das ablações (aprovada em 18/10/2026, antes de qualquer ablação; docs/08-ablacoes.md).
# Cada ablação roda com as 3 sementes e é comparada pela média, só na validação, com a v2.
# Aceita se a AUC subir pelo menos ABLATION_MIN_AUC_GAIN ou a localização (pixel mais
# quente dentro da caixa, medida acima do acaso de cada configuração) subir pelo menos
# ABLATION_MIN_POINTING_GAIN, desde que a outra não piore mais que um desvio-padrão da v2
# entre sementes.
ABLATION_MIN_AUC_GAIN = 0.02
ABLATION_MIN_POINTING_GAIN = 0.08
ABLATION_REFERENCE = ["v2-linha-de-base", "v2-semente-1", "v2-semente-2"]

# Hiperparâmetros
LATENT_DIM = 8  # tamanho do código latente z
BETA = 1.0  # peso do termo KL (β = 1 → VAE clássico)
EPOCHS = 40
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
KL_WARMUP_EPOCHS = 10  # o peso do KL cresce de 0 até β nas primeiras épocas
SEEDS = [0, 1, 2]  # o VAE e o autoencoder comum são treinados com 3 sementes

# Score de anomalia
# Larguras testadas na suavização do resíduo, em pixels de uma imagem de 64 px. Em outras
# resoluções elas são escaladas, para cobrir as mesmas estruturas anatômicas. A escolhida
# é a de maior AUC na validação.
SIGMAS = [0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8]
# Limiar sem rótulos: o percentil 95 dos normais da validação aceita ~5% de falsos alarmes.
THRESHOLD_PERCENTILE = 95


@dataclass(frozen=True)
class RunPaths:
    """Onde cada execução do pipeline guarda modelo e relatórios.

    Cada combinação de incidências e versão vira uma pasta própria (models/rsna_pa_v2 e
    reports/rsna_pa_v2, ou rsna_ap_pa_v2), para os resultados não se sobrescreverem.
    """

    name: str

    @property
    def exists(self) -> bool:
        """Se a execução já gravou modelo ou relatórios."""
        return (MODELS_DIR / self.name).exists() or self.reports.exists()

    @property
    def model(self) -> Path:
        return MODELS_DIR / self.name / "vae.pt"

    @property
    def metadata(self) -> Path:
        return MODELS_DIR / self.name / "metadata.json"

    @property
    def reports(self) -> Path:
        return REPORTS_DIR / self.name

    @property
    def figures(self) -> Path:
        return REPORTS_DIR / self.name / "figures"


def run_name(views: list[str], version: str | None = None) -> str:
    """Nome da execução: "rsna_pa" ou "rsna_ap_pa", com o sufixo da versão.

    A v1 (e version=None) não tem sufixo: são as pastas da primeira versão do pipeline e o
    nome dos experimentos no MLflow, que reúnem todas as versões.
    """
    name = "rsna_" + "_".join(sorted(v.lower() for v in views))
    return name if version in (None, "v1") else f"{name}_{version}"


def get_device() -> torch.device:
    """Escolhe a GPU disponível (CUDA ou Apple MPS) e cai para a CPU se não houver."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
