from deepface import DeepFace
from deepface.modules.exceptions import FaceNotDetected
import numpy as np


def extract_faces(image: np.ndarray, model_name: str, detector_backend: str) -> list[dict]:
    """
    Encontra os rostos da imagem e gera o embedding de cada um.

    O embedding é um vetor de números que resume o rosto. Fotos da mesma
    pessoa geram vetores próximos; pessoas diferentes geram vetores distantes.
    É essa a feature usada para comparar rostos.

    Args:
        image (np.ndarray): Imagem em BGR (foto do disco ou frame da webcam).
        model_name (str): Modelo do DeepFace que gera o embedding.
        detector_backend (str): Detector do DeepFace que localiza os rostos.

    Returns:
        list[dict]: Um item por rosto, com as chaves:
            - embedding: vetor do rosto (np.ndarray).
            - facial_area: posição do rosto na imagem (x, y, w, h).
            Lista vazia quando não há rosto.
    """
    try:
        faces = DeepFace.represent(
            img_path=image,
            model_name=model_name,
            detector_backend=detector_backend,
            # Com True, uma imagem sem rosto gera FaceNotDetected.
            # Com False, o DeepFace usaria a imagem inteira como se fosse um rosto.
            enforce_detection=True,
        )
    except FaceNotDetected:
        return []

    return [
        {
            "embedding": np.asarray(face["embedding"], dtype=np.float32),
            "facial_area": face["facial_area"],
        }
        for face in faces
    ]
