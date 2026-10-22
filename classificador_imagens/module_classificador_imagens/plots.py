"""Gráficos do projeto, salvos em reports/figures."""

from loguru import logger
import matplotlib.pyplot as plt


def save_figure(fig, path) -> None:
    """
    Salva a figura em disco e fecha para liberar memória.

    Args:
        fig (plt.Figure): figura a ser salva.
        path (Path): arquivo de destino (.png).
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    logger.success(f"Figura salva em: {path}")


def plot_image_grid(images, titles, suptitle, path) -> None:
    """
    Monta uma grade 3x3 de imagens, cada uma com seu título.

    Args:
        images (np.ndarray): até 9 imagens.
        titles (list[str]): um título para cada imagem.
        suptitle (str): título geral da figura.
        path (Path): arquivo de destino.
    """
    fig = plt.figure(figsize=(10, 10))

    for i, (image, title) in enumerate(zip(images, titles)):
        ax = fig.add_subplot(3, 3, i + 1)
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(suptitle)
    save_figure(fig, path)


def plot_sample_images(X, y, class_names, path) -> None:
    """
    Mostra as 9 primeiras imagens da base com o nome da classe.

    Args:
        X (np.ndarray): imagens.
        y (np.ndarray): rótulos.
        class_names (list[str]): nome de cada classe.
        path (Path): arquivo de destino.
    """
    plot_image_grid(
        images=X[:9],
        titles=[class_names[label] for label in y[:9]],
        suptitle="Exemplos da base CIFAR-10",
        path=path,
    )


def plot_training_history(history, path) -> None:
    """
    Compara acurácia e loss de treino e validação ao longo das épocas.

    Args:
        history (dict | pd.DataFrame): histórico do fit, com as chaves
            accuracy, val_accuracy, loss e val_loss.
        path (Path): arquivo de destino.
    """
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(12, 4))

    # Acurácia por época
    ax_acc.plot(history["accuracy"], label="Treino")
    ax_acc.plot(history["val_accuracy"], label="Validação")
    ax_acc.set_title("Acurácia por Época")
    ax_acc.set_xlabel("Épocas")
    ax_acc.set_ylabel("Acurácia")
    ax_acc.legend()

    # Loss por época
    ax_loss.plot(history["loss"], label="Treino")
    ax_loss.plot(history["val_loss"], label="Validação")
    ax_loss.set_title("Loss por Época")
    ax_loss.set_xlabel("Épocas")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()

    save_figure(fig, path)


def plot_predictions(X, y_true, y_pred, class_names, path) -> None:
    """
    Mostra até 9 imagens com a classe real e a classe prevista.

    Args:
        X (np.ndarray): imagens.
        y_true (np.ndarray): rótulos reais.
        y_pred (np.ndarray): rótulos previstos.
        class_names (list[str]): nome de cada classe.
        path (Path): arquivo de destino.
    """
    plot_image_grid(
        images=X[:9],
        titles=[
            f"Real: {class_names[real]}\nPredito: {class_names[pred]}"
            for real, pred in zip(y_true[:9], y_pred[:9])
        ],
        suptitle="Previsões do Modelo",
        path=path,
    )
