"""Inferência com o modelo salvo por `make train`.

    uv run python -m module_vae.inference                    # 5 exames sorteados do teste
    uv run python -m module_vae.inference --image exame.png  # um raio-X qualquer
    uv run python -m module_vae.inference --run rsna_ap_pa       # o modelo treinado com AP e PA

A imagem externa é convertida para tons de cinza e redimensionada para a resolução do
modelo. Ela deve ser um raio-X de tórax frontal inteiro, como os do RSNA: exames com
outro enquadramento tendem a receber score alto. O modelo
principal foi treinado só com incidência PA; um exame AP também tende a receber score
alto.
"""

from pathlib import Path
from typing import Annotated

from loguru import logger
import numpy as np
from PIL import Image
import torch
import typer

from module_vae.config import RunPaths
from module_vae.modeling.predict import load_model, predict
from module_vae.modeling.split import load_splits
from module_vae.plots import apply_style, plot_heatmaps
from module_vae.preprocessing import Preprocessing

app = typer.Typer()


def load_image(
    path: Path, image_size: int, preprocessing: Preprocessing | None = None
) -> torch.Tensor:
    """Lê uma imagem e devolve um tensor (1, 1, H, W) com pixels em [0, 1].

    preprocessing aplica as mesmas transformações (recorte, normalização) do treino.
    """
    img = Image.open(path).convert("L")
    img = np.asarray(img.resize((image_size, image_size), Image.Resampling.BILINEAR))
    if preprocessing is not None:
        img = preprocessing.apply(img[None])[0]
    return torch.tensor(img, dtype=torch.float32)[None, None] / 255.0


@app.command()
def main(
    image: Annotated[
        Path | None, typer.Option(help="Raio-X (png/jpg) a avaliar. Sem ele, usa o teste.")
    ] = None,
    n_samples: Annotated[
        int, typer.Option(help="Exames sorteados do teste quando não há --image.")
    ] = 5,
    run: Annotated[
        str, typer.Option(help="Execução treinada: rsna_pa, rsna_ap_pa, rsna_pa_v2...")
    ] = "rsna_pa",
):
    paths = RunPaths(run)
    model, metadata = load_model(paths.model, paths.metadata)
    preprocessing = Preprocessing.from_metadata(metadata)

    if image is not None:
        x, labels = load_image(image, metadata["image_size"], preprocessing), None
        title = image.name
    else:
        splits = load_splits(
            metadata["image_size"],
            metadata["views"],
            # modelos da v1 não têm "hygiene" nos metadados: foram treinados sem higiene
            hygiene=metadata.get("hygiene", False),
            preprocessing=preprocessing,
        )
        idx = np.random.default_rng(42).choice(len(splits.x_test), n_samples, replace=False)
        x = splits.x_test[idx]
        labels = np.where(splits.y_test[idx] == 1, metadata["positive_class"], "normal")
        title = f"{n_samples} exames sorteados do teste"

    predictions, x_rec, maps = predict(
        model, x, metadata["sigma"], metadata["threshold"], metadata.get("mask")
    )

    # queue_position ordena só os exames avaliados nesta chamada.
    predictions["decision"] = predictions["prediction"].map(
        {1: "revisar com prioridade", 0: "dentro do padrão"}
    )
    if labels is not None:
        predictions.insert(0, "label", labels)
    logger.success(f"Predições:\n{predictions}")

    apply_style()
    plot_heatmaps(
        x,
        x_rec,
        maps,
        predictions["anomaly_score"].to_numpy(),
        list(range(len(x))),
        metadata["vmax_map"],
        title,
        paths.figures / "inference.png",
        labels=labels,
    )


if __name__ == "__main__":
    app()
