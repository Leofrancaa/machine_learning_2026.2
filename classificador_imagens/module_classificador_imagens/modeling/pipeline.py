"""Arquitetura da CNN usada para classificar as imagens do CIFAR-10."""

from tensorflow import keras
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
    RandomFlip,
    RandomTranslation,
)

from module_classificador_imagens.dataset import CLASS_NAMES

MODEL_NAME = "CNN"

# Cada imagem tem 32x32 pixels e 3 canais de cor (RGB)
INPUT_SHAPE = (32, 32, 3)

# Uma saída para cada classe do CIFAR-10
NUM_CLASSES = len(CLASS_NAMES)


def create_augmentation() -> keras.Sequential:
    """
    Cria o bloco de data augmentation.

    Essas camadas geram pequenas variações das imagens a cada época, mas só
    durante o treino. Na avaliação e na predição a imagem passa sem alteração.

    Returns:
        keras.Sequential: camadas de data augmentation.
    """
    return keras.Sequential(
        [
            # Espelha a imagem na horizontal de forma aleatória
            RandomFlip("horizontal"),
            # Desloca a imagem em até 10% na altura e na largura
            RandomTranslation(height_factor=0.1, width_factor=0.1),
        ],
        name="data_augmentation",
    )


def add_conv_block(model: keras.Sequential, filters: int, dropout: float) -> None:
    """
    Adiciona um bloco convolucional ao modelo.

    O bloco tem duas convoluções (cada uma seguida de BatchNormalization),
    um MaxPooling que reduz a imagem pela metade e um Dropout no final.

    Args:
        model (keras.Sequential): modelo que vai receber o bloco.
        filters (int): quantidade de filtros das convoluções.
        dropout (float): fração de neurônios desligados no treino.
    """
    # padding="same" mantém o tamanho da imagem depois da convolução
    model.add(Conv2D(filters=filters, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=filters, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization())

    # Reduz a imagem pela metade: 32 -> 16 -> 8 -> 4 ao longo dos três blocos
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))


def create_cnn_model() -> keras.Sequential:
    """
    Monta e compila a CNN.

    Returns:
        keras.Sequential: modelo compilado, pronto para o treino.
    """
    model = keras.Sequential(name="cnn")

    # Entrada: imagens 32x32 com 3 canais
    model.add(Input(shape=INPUT_SHAPE))

    # Variações aleatórias das imagens (só no treino)
    model.add(create_augmentation())

    # Três blocos convolucionais: os filtros dobram a cada bloco e o dropout
    # aumenta para conter o overfitting
    add_conv_block(model, filters=32, dropout=0.2)
    add_conv_block(model, filters=64, dropout=0.3)
    add_conv_block(model, filters=128, dropout=0.4)

    # Transforma os mapas de características em um vetor e classifica
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))

    # Uma saída por classe; o softmax transforma as saídas em probabilidades
    model.add(Dense(NUM_CLASSES, activation="softmax"))

    # Rótulos inteiros (0 a 9) -> sparse_categorical_crossentropy
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
