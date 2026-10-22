import json

from deepface.modules.verification import find_threshold
from loguru import logger
import pandas as pd
from tqdm import tqdm

from module_reconhecimento.dataset import read_image
from module_reconhecimento.features import extract_faces


def build_face_database(
    images,
    model_name,
    detector_backend,
    distance_metric,
    threshold,
    database_path,
    metadata_path,
):
    """
    Gera o banco de rostos autorizados e salva o banco e os metadados.

    O modelo do DeepFace já vem treinado, então nenhum peso é ajustado aqui.
    O "treino" é o cadastro: gerar o embedding de cada foto autorizada para
    depois comparar com os rostos que aparecem no acesso.

    Args:
        images (pd.DataFrame): Saída do load_authorized_images (name, image_path).
        model_name (str): Modelo do DeepFace que gera os embeddings.
        detector_backend (str): Detector do DeepFace que localiza os rostos.
        distance_metric (str): Métrica usada para comparar embeddings.
        threshold (float | None): Distância máxima para liberar o acesso.
            None usa o valor calibrado do DeepFace.
        database_path (Path): Onde salvar o banco de embeddings (.pkl).
        metadata_path (Path): Onde salvar os metadados (.json).

    Returns:
        pd.DataFrame: Banco com as colunas name, image_path e embedding.
    """
    logger.info(f"Gerando embeddings com {model_name} (detector: {detector_backend})...")

    rows = []

    for row in tqdm(images.itertuples(), total=len(images)):
        try:
            image = read_image(row.image_path)
        except ValueError as error:
            logger.warning(f"{error}. Foto ignorada.")
            continue

        faces = extract_faces(
            image,
            model_name=model_name,
            detector_backend=detector_backend,
        )

        if not faces:
            logger.warning(f"Nenhum rosto encontrado em {row.image_path}. Foto ignorada.")
            continue

        # A foto de cadastro deveria ter só a pessoa. Se aparecer mais de um
        # rosto (alguém ao fundo, por exemplo), fica o maior, que é o principal.
        if len(faces) > 1:
            logger.warning(f"{len(faces)} rostos em {row.image_path}. Usando o maior.")

        face = max(faces, key=lambda face: face["facial_area"]["w"] * face["facial_area"]["h"])

        rows.append(
            {
                "name": row.name,
                "image_path": str(row.image_path),
                "embedding": face["embedding"],
            }
        )

    if not rows:
        raise ValueError("Nenhum rosto cadastrado. Confira as fotos das pessoas autorizadas.")

    database = pd.DataFrame(rows)

    # -------------------------------------------------
    # Threshold: distância máxima para liberar o acesso
    # -------------------------------------------------

    if threshold is None:
        threshold = find_threshold(model_name, distance_metric)

    logger.info(f"Threshold: {threshold:.2f}")

    # -------------------------------------------------
    # Salva o banco de embeddings
    # -------------------------------------------------

    database_path.parent.mkdir(parents=True, exist_ok=True)
    database.to_pickle(database_path)

    logger.success(f"Banco de rostos salvo em: {database_path}")

    # -------------------------------------------------
    # Salva os metadados
    # -------------------------------------------------

    # O acesso precisa usar o mesmo modelo, detector e métrica do cadastro:
    # embeddings de modelos diferentes não podem ser comparados.
    metadata = {
        "model_name": model_name,
        "detector_backend": detector_backend,
        "distance_metric": distance_metric,
        "threshold": float(threshold),
        "n_people": int(database["name"].nunique()),
        "n_images": len(database),
    }

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    logger.success(f"Metadados salvos em: {metadata_path}")
    logger.info(f"Rostos cadastrados por pessoa:\n{database['name'].value_counts()}")

    return database
