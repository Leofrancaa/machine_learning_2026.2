from loguru import logger

from module_reconhecimento.config import (
    AUTHORIZED_DIR,
    DETECTOR_BACKEND,
    DISTANCE_METRIC,
    MODEL_NAME,
    MODELS_DIR,
    THRESHOLD,
)
from module_reconhecimento.dataset import load_authorized_images
from module_reconhecimento.modeling.train import build_face_database


def main():

    logger.info("Carregando as fotos das pessoas autorizadas...")

    images = load_authorized_images(AUTHORIZED_DIR)

    # =============================================
    # CADASTRO
    # Gera o embedding de cada foto autorizada
    # =============================================

    build_face_database(
        images=images,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR_BACKEND,
        distance_metric=DISTANCE_METRIC,
        threshold=THRESHOLD,
        database_path=MODELS_DIR / "face_database.pkl",
        metadata_path=MODELS_DIR / "metadata.json",
    )

    logger.success("Cadastro executado com sucesso.")


if __name__ == "__main__":
    main()
