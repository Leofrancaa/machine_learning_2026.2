"""Download e carregamento do RSNA Pneumonia Detection Challenge (Kaggle).

São 26.684 raios-X de tórax de adultos com rótulos, um por paciente, em DICOM de 1024 px.
Cada exame tem uma de três classes (normal, opacidade pulmonar ou anormal sem opacidade),
e as opacidades vêm marcadas com caixas. Os DICOMs são lidos uma vez e guardados
redimensionados em data/processed.

A partir da v2, os metadados de cada exame (idade, sexo e PixelSpacing, que identifica o
equipamento) vêm dos cabeçalhos dos DICOMs, guardados à parte em rsna_headers.csv. Assim
o cache das imagens da v1 continua o mesmo.
"""

import hashlib
from pathlib import Path

import kagglehub
from loguru import logger
import numpy as np
import pandas as pd
from PIL import Image
import pydicom
from tqdm import tqdm
import typer

from module_vae.config import IMAGE_SIZE, MAX_VALID_AGE, MIN_AGE, PROCESSED_DATA_DIR, RSNA_DIR

RSNA_COMPETITION = "rsna-pneumonia-detection-challenge"
RSNA_ORIGINAL_SIZE = 1024
RSNA_CLASSES = {
    "Normal": "normal",
    "Lung Opacity": "opacity",
    "No Lung Opacity / Not Normal": "other",
}

app = typer.Typer()


def download_rsna(root: Path = RSNA_DIR) -> None:
    """Baixa os arquivos da competição do Kaggle, se ainda não estiverem em root.

    Precisa de um token da API do Kaggle (~/.kaggle/access_token ou kaggle.json) e de as
    regras da competição terem sido aceitas no site.
    """
    if (root / "stage_2_train_labels.csv").exists():
        return

    logger.info(f"Baixando {RSNA_COMPETITION} do Kaggle para {root}...")
    kagglehub.competition_download(RSNA_COMPETITION, output_dir=str(root))


def read_dicom(path: Path, image_size: int) -> tuple[np.ndarray, str]:
    """Lê um DICOM e devolve a imagem redimensionada (uint8) e a incidência (AP ou PA)."""
    ds = pydicom.dcmread(path)
    img = ds.pixel_array

    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        img = (255 * (img - img.min()) / max(img.max() - img.min(), 1)).astype(np.uint8)
    # Em MONOCHROME1 o osso aparece escuro; invertendo, todos ficam com osso claro.
    if ds.PhotometricInterpretation == "MONOCHROME1":
        img = 255 - img

    img = Image.fromarray(img).resize((image_size, image_size), Image.Resampling.BILINEAR)
    return np.asarray(img), ds.ViewPosition


def rsna_cache_path(image_size: int) -> Path:
    return PROCESSED_DATA_DIR / f"rsna_{image_size}.npz"


def build_rsna_cache(image_size: int, root: Path = RSNA_DIR) -> Path:
    """Lê os 26.684 DICOMs de treino uma vez e guarda as imagens redimensionadas.

    Só o conjunto de treino da competição tem rótulos; as imagens de teste dela não são
    usadas. Os splits de treino, validação e teste da PoC saem desse conjunto.
    """
    download_rsna(root)

    # Há uma linha por caixa; a classe é a mesma em todas as linhas de um paciente.
    meta = pd.read_csv(root / "stage_2_detailed_class_info.csv").drop_duplicates("patientId")

    images, views = [], []
    for patient_id in tqdm(meta["patientId"], desc=f"DICOM → {image_size} px"):
        img, view = read_dicom(root / "stage_2_train_images" / f"{patient_id}.dcm", image_size)
        images.append(img)
        views.append(view)

    path = rsna_cache_path(image_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        images=np.stack(images),
        patient_ids=meta["patientId"].to_numpy().astype(str),
        labels=meta["class"].map(RSNA_CLASSES).to_numpy().astype(str),
        views=np.array(views),
    )
    logger.success(f"Cache do RSNA salvo em {path}")

    return path


def rsna_headers_path() -> Path:
    return PROCESSED_DATA_DIR / "rsna_headers.csv"


def read_header(path: Path) -> dict:
    """Idade, sexo e PixelSpacing de um DICOM, sem ler os pixels.

    O PixelSpacing não serve como escala física (a imagem foi reduzida para 1024 px depois
    de medido), mas cada valor corresponde a um equipamento (docs/06-eda-rsna.md).
    """
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    return {
        "age": int(ds.PatientAge),
        "sex": str(ds.PatientSex),
        "pixel_spacing": round(float(ds.PixelSpacing[0]), 3),
    }


def build_rsna_headers(root: Path = RSNA_DIR) -> Path:
    """Lê os cabeçalhos dos 26.684 DICOMs de treino e guarda idade, sexo e PixelSpacing."""
    download_rsna(root)
    patient_ids = pd.read_csv(root / "stage_2_detailed_class_info.csv")["patientId"].unique()

    rows = [
        {"patientId": pid, **read_header(root / "stage_2_train_images" / f"{pid}.dcm")}
        for pid in tqdm(patient_ids, desc="Cabeçalhos DICOM")
    ]

    path = rsna_headers_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    logger.success(f"Cabeçalhos do RSNA salvos em {path}")

    return path


def image_hashes(images: np.ndarray) -> np.ndarray:
    """Hash dos pixels de cada imagem: imagens idênticas têm o mesmo hash."""
    return np.array([hashlib.sha1(img.tobytes()).hexdigest() for img in images])


def exclusion_reasons(images: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
    """Motivo para tirar cada exame do escopo da PoC ("" = o exame fica).

    - "duplicata": imagem com pixels idênticos aos de outro exame (o mesmo exame
      cadastrado com dois patientIds). Fica o de menor patientId.
    - "idade ausente": idade acima de MAX_VALID_AGE (impossível) ou desconhecida. Sem a
      idade, não dá para confirmar que é um adulto.
    - "menor de 18": fora do escopo de adultos.

    Cada exame recebe só o primeiro motivo, nessa ordem.

    Args:
        images (np.ndarray): Imagens (N, H, W).
        meta (pd.DataFrame): patientId e age de cada imagem, na mesma ordem.
    """
    hashes = pd.Series(image_hashes(images), index=meta.index)
    first = meta.assign(hash=hashes).sort_values("patientId").drop_duplicates("hash")
    duplicate = ~meta.index.isin(first.index)

    age = meta["age"].where(meta["age"] <= MAX_VALID_AGE)
    reasons = np.select(
        [duplicate, age.isna().to_numpy(), (age < MIN_AGE).to_numpy()],
        ["duplicata", "idade ausente", f"menor de {MIN_AGE}"],
        default="",
    )
    return reasons


def load_rsna(
    image_size: int = IMAGE_SIZE,
    root: Path = RSNA_DIR,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Carrega o RSNA redimensionado, criando o cache na primeira vez.

    Returns:
        tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
            - images: (N, H, W) em uint8;
            - meta: uma linha por exame, na mesma ordem de images, com patientId, label
              ("normal", "opacity" ou "other"), view ("AP" ou "PA"), age (idades acima
              de MAX_VALID_AGE viram ausentes), sex, pixel_spacing (o equipamento) e
              exclusion (motivo para ficar fora do escopo, "" se fica; ver
              exclusion_reasons);
            - boxes: caixas das opacidades (patientId, x, y, width, height), já em pixels
              da imagem redimensionada.
    """
    path = rsna_cache_path(image_size)
    if not path.exists():
        build_rsna_cache(image_size, root)
    if not rsna_headers_path().exists():
        build_rsna_headers(root)

    cache = np.load(path)
    meta = pd.DataFrame(
        {
            "patientId": cache["patient_ids"],
            "label": cache["labels"],
            "view": cache["views"],
        }
    )
    headers = pd.read_csv(rsna_headers_path())
    meta = meta.merge(headers, on="patientId", how="left", validate="one_to_one")
    meta["exclusion"] = exclusion_reasons(cache["images"], meta)
    meta["age"] = meta["age"].where(meta["age"] <= MAX_VALID_AGE)

    boxes = pd.read_csv(root / "stage_2_train_labels.csv").dropna(subset=["x"])
    scale = image_size / RSNA_ORIGINAL_SIZE
    boxes[["x", "y", "width", "height"]] *= scale

    logger.info(f"RSNA carregado: {cache['images'].shape} | {len(boxes)} caixas")
    logger.info(
        f"Fora do escopo: {meta['exclusion'].replace('', 'no escopo').value_counts().to_dict()}"
    )
    return cache["images"], meta, boxes[["patientId", "x", "y", "width", "height"]]


@app.command()
def main():
    """Baixa o RSNA e cria o cache das imagens redimensionadas."""
    _, meta, _ = load_rsna()
    logger.success(f"Exames por classe e incidência:\n{pd.crosstab(meta.label, meta.view)}")


if __name__ == "__main__":
    app()
