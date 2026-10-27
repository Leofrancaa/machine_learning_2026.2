"""Montagem dos conjuntos: o ponto central do desenho experimental.

O VAE só vê exames normais no treino. Os exames alterados do treino ficam de fora (só a
CNN supervisionada, usada como referência, os utiliza).

- Treino do VAE: os exames normais do split de treino.
- Validação: acompanhar o treino (normais), calibrar o limiar (normais) e escolher σ
  (normais + positivos).
- Teste: reservado para a avaliação final.

O "positivo" é a opacidade pulmonar (pneumonia). Os exames "anormais sem opacidade" do
teste ficam à parte (x_test_other): eles medem se o VAE sinaliza alterações que não são
pneumonia, sem terem sido usados para escolher nada.

A partir da v2, os exames fora do escopo (duplicata, idade ausente, menores de 18) saem
dos três conjuntos. Assim como o filtro de incidência, isso acontece depois do sorteio:
o split é o mesmo da v1, e o teste da v2 é o teste da v1 sem esses exames.
"""

from dataclasses import dataclass

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

from module_vae.dataset import load_rsna
from module_vae.preprocessing import Preprocessing


@dataclass
class DataSplits:
    x_train: torch.Tensor
    y_train: np.ndarray
    x_val: torch.Tensor
    y_val: np.ndarray
    x_test: torch.Tensor
    y_test: np.ndarray
    x_train_normal: torch.Tensor
    x_val_normal: torch.Tensor
    x_test_other: torch.Tensor  # anormais sem opacidade
    boxes_val: pd.DataFrame  # caixas das opacidades (exam_index, x, y, width, height)
    boxes_test: pd.DataFrame
    views_val: np.ndarray  # incidência (AP ou PA) de cada exame de x_val
    views_test: np.ndarray
    equipment_val: np.ndarray  # PixelSpacing de cada exame de x_val, como texto ("0.143")
    equipment_test: np.ndarray
    patient_ids_test: np.ndarray  # patientId de cada exame de x_test (para achar o DICOM)
    patient_ids_test_other: np.ndarray
    excluded: pd.DataFrame  # exames fora do escopo, por conjunto (linhas) e motivo


def split_rsna(
    images: np.ndarray,
    meta: pd.DataFrame,
    boxes: pd.DataFrame,
    views: list[str],
    seed: int = 42,
    hygiene: bool = True,
) -> DataSplits:
    """Divide o RSNA em 70% treino, 15% validação e 15% teste.

    Args:
        images (np.ndarray): Imagens (N, H, W) em uint8.
        meta (pd.DataFrame): patientId, label, view, pixel_spacing e exclusion de cada
            imagem (ver dataset.load_rsna).
        boxes (pd.DataFrame): Caixas das opacidades (patientId, x, y, width, height).
        views (list[str]): Incidências mantidas ("PA", ou "AP" e "PA").
        seed (int): Semente do sorteio.
        hygiene (bool): Tira os exames fora do escopo (coluna exclusion). False reproduz
            os conjuntos da v1.

    Returns:
        DataSplits: Conjuntos com normais (0) e opacidades (1), mais os anormais sem
            opacidade, as caixas e as incidências do teste.
    """
    # Há um exame por paciente, então separar exames é o mesmo que separar pacientes:
    # ninguém aparece em dois conjuntos. A estratificação mantém a proporção de cada
    # classe e incidência nos três conjuntos.
    strata = (meta["label"] + "_" + meta["view"]).to_numpy()
    idx = np.arange(len(meta))
    train_idx, rest_idx = train_test_split(idx, test_size=0.30, stratify=strata, random_state=seed)
    val_idx, test_idx = train_test_split(
        rest_idx, test_size=0.50, stratify=strata[rest_idx], random_state=seed
    )

    # Os filtros de incidência e de escopo vêm depois do sorteio: assim o teste só com PA
    # é um pedaço do teste com AP e PA, o teste da v2 é um pedaço do teste da v1, e os
    # experimentos podem ser comparados.
    labels = meta["label"].to_numpy()
    in_views = meta["view"].isin(views).to_numpy()
    exclusion = meta["exclusion"].to_numpy() if hygiene else np.full(len(meta), "")
    in_scope = exclusion == ""

    def select(split_idx: np.ndarray, classes: list[str]) -> np.ndarray:
        keep = in_views[split_idx] & in_scope[split_idx] & np.isin(labels[split_idx], classes)
        return split_idx[keep]

    def to_tensor(ix: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(images[ix]).float().unsqueeze(1) / 255.0

    train_ix = select(train_idx, ["normal", "opacity"])
    val_ix = select(val_idx, ["normal", "opacity"])
    test_ix = select(test_idx, ["normal", "opacity"])
    other_ix = select(test_idx, ["other"])

    y_train = (labels[train_ix] == "opacity").astype(int)
    y_val = (labels[val_ix] == "opacity").astype(int)
    y_test = (labels[test_ix] == "opacity").astype(int)
    x_train, x_val = to_tensor(train_ix), to_tensor(val_ix)

    def boxes_of(ix: np.ndarray) -> pd.DataFrame:
        """Caixas das opacidades de um conjunto, com a posição de cada exame nele."""
        position = {pid: i for i, pid in enumerate(meta["patientId"].to_numpy()[ix])}
        return boxes[boxes["patientId"].isin(position)].assign(
            exam_index=lambda df: df["patientId"].map(position)
        )

    def equipment_of(ix: np.ndarray) -> np.ndarray:
        return meta["pixel_spacing"].to_numpy()[ix].astype(str)

    boxes_val, boxes_test = boxes_of(val_ix), boxes_of(test_ix)

    sets = [("treino", train_idx), ("validação", val_idx), ("teste", test_idx)]
    counts = pd.DataFrame(
        {
            name: pd.Series(labels[select(ix, list(set(labels)))]).value_counts()
            for name, ix in sets
        }
    ).T
    # Exames que seriam usados (incidência certa) mas estão fora do escopo
    excluded = (
        pd.DataFrame(
            {
                name: pd.Series(exclusion[ix[in_views[ix] & ~in_scope[ix]]]).value_counts()
                for name, ix in sets
            }
        )
        .T.fillna(0)
        .astype(int)
    )
    logger.info(f"RSNA, incidências {views}:\n{counts}")
    if hygiene:
        logger.info(f"Fora do escopo (já descontados acima):\n{excluded}")
    logger.info(
        f"Caixas: validação {len(boxes_val)} em {boxes_val['exam_index'].nunique()} exames, "
        f"teste {len(boxes_test)} em {boxes_test['exam_index'].nunique()} exames"
    )

    return DataSplits(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=to_tensor(test_ix),
        y_test=y_test,
        x_train_normal=x_train[torch.from_numpy(y_train == 0)],
        x_val_normal=x_val[torch.from_numpy(y_val == 0)],
        x_test_other=to_tensor(other_ix),
        boxes_val=boxes_val,
        boxes_test=boxes_test,
        views_val=meta["view"].to_numpy()[val_ix],
        views_test=meta["view"].to_numpy()[test_ix],
        equipment_val=equipment_of(val_ix),
        equipment_test=equipment_of(test_ix),
        patient_ids_test=meta["patientId"].to_numpy()[test_ix],
        patient_ids_test_other=meta["patientId"].to_numpy()[other_ix],
        excluded=excluded,
    )


def load_splits(
    image_size: int,
    views: list[str],
    hygiene: bool = True,
    preprocessing: Preprocessing | None = None,
) -> DataSplits:
    """Carrega o RSNA na resolução pedida, aplica as ablações A e B e monta os conjuntos."""
    images, meta, boxes = load_rsna(image_size)
    if preprocessing is not None:
        images = preprocessing.apply(images)
    return split_rsna(images, meta, boxes, views, hygiene=hygiene)
