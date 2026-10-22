"""Avaliação final do modelo no conjunto de teste."""

from loguru import logger


def evaluate_model(model, model_name, X_test, y_test) -> dict:
    """
    Avalia o modelo em imagens que ele nunca viu no treino.

    Args:
        model (keras.Model): modelo treinado.
        model_name (str): nome do modelo, usado no log.
        X_test (np.ndarray): imagens de teste já normalizadas.
        y_test (np.ndarray): rótulos de teste.

    Returns:
        dict: nome do modelo, loss e acurácia no teste.
    """
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    logger.info("=" * 60)
    logger.success(f"MODELO FINAL: {model_name}")
    logger.info(f"Loss no teste: {test_loss:.3f}")
    logger.info(f"Acurácia no teste: {test_accuracy:.3f}")

    return {
        "model_name": model_name,
        "loss": test_loss,
        "accuracy": test_accuracy,
    }
