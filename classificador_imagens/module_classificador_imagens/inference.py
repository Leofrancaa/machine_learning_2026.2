"""Inferência com o modelo já treinado em algumas imagens do teste.

Execute na pasta classificador_imagens (depois do main.py) com:
    uv run python -m module_classificador_imagens.inference
"""

from loguru import logger
import numpy as np

from module_classificador_imagens.config import FIGURES_DIR, MODELS_DIR
from module_classificador_imagens.dataset import load_data
from module_classificador_imagens.features import normalize_images
from module_classificador_imagens.modeling.predict import load_model, predict
from module_classificador_imagens.plots import plot_predictions

# Quantidade de imagens usadas na inferência (cabe na grade 3x3 do gráfico)
N_SAMPLES = 9


def main():
    # Só o conjunto de teste interessa aqui: são imagens que o modelo nunca viu
    _, X_test, _, y_test = load_data()

    # Sorteia algumas imagens (semente fixa para sortear sempre as mesmas)
    rng = np.random.default_rng(42)
    indices = rng.choice(len(X_test), size=N_SAMPLES, replace=False)

    # Mesma normalização usada no treino
    X_sample = normalize_images(X_test[indices])
    y_sample = y_test[indices]

    # Carrega o modelo e os nomes das classes salvos pelo main.py
    model, _, class_names = load_model(
        model_path=MODELS_DIR / "cnn_model.keras",
        metadata_path=MODELS_DIR / "metadata.json",
    )

    predictions = predict(model=model, X=X_sample, class_names=class_names)

    # Acrescenta a posição da imagem no teste e a classe real, para comparar
    predictions.index = indices
    predictions.index.name = "test_index"
    predictions.insert(0, "real", [class_names[label] for label in y_sample])

    logger.success(f"Predições:\n{predictions}")

    plot_predictions(
        X_sample,
        y_sample,
        predictions["prediction"].to_numpy(),
        class_names,
        FIGURES_DIR / "predictions.png",
    )


if __name__ == "__main__":
    main()
