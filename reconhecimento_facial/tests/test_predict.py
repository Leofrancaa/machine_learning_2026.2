import numpy as np
import pandas as pd

from module_reconhecimento.modeling.predict import UNKNOWN_NAME, identify_faces

# Banco com dois "rostos" em direções perpendiculares. Na distância cosseno
# só importa o ângulo entre os vetores.
DATABASE = pd.DataFrame(
    {
        "name": ["maria", "joao"],
        "embedding": [np.array([1.0, 0.0]), np.array([0.0, 1.0])],
    }
)


def test_each_face_gets_the_closest_person():
    predictions = identify_faces(
        embeddings=np.array([[0.9, 0.1], [0.1, 0.9]]),
        database=DATABASE,
        distance_metric="cosine",
        threshold=0.30,
    )

    assert predictions["name"].tolist() == ["maria", "joao"]
    assert predictions["authorized"].all()


def test_face_far_from_everyone_is_denied():
    predictions = identify_faces(
        embeddings=np.array([[-1.0, -1.0]]),
        database=DATABASE,
        distance_metric="cosine",
        threshold=0.30,
    )

    assert predictions.loc[0, "name"] == UNKNOWN_NAME
    assert not predictions.loc[0, "authorized"]


def test_threshold_decides_the_access():
    # [1, 1] está a 45° dos dois rostos: distância = 1 - cos(45°) ≈ 0.29.
    embeddings = np.array([[1.0, 1.0]])

    loose = identify_faces(embeddings, DATABASE, "cosine", threshold=0.30)
    strict = identify_faces(embeddings, DATABASE, "cosine", threshold=0.20)

    assert loose.loc[0, "authorized"]
    assert not strict.loc[0, "authorized"]
