"""Exporta os dados da demo (interfaces/) a partir dos modelos treinados.

    uv run python -m module_vae.export                      # rsna_pa_v3 e rsna_ap_pa_v3
    uv run python -m module_vae.export --shortcut-run ""    # só o modelo PA

O site (Next.js, em interfaces/) não roda o modelo: scores, reconstruções, mapas de
calor, o espaço latente e as imagens do gerador são calculados aqui e gravados em
interfaces/public/data, junto com o data.json que a página lê.

- O modelo principal (só PA) alimenta as áreas da demonstração.
- O modelo com AP e PA (opcional) só alimenta a fila, no botão "AP + PA", para mostrar
  o atalho da incidência.

Todos os exames vêm do conjunto de teste, que não foi usado para escolher nada. As
imagens são do RSNA: veja a atribuição exigida em interfaces/README.md.

Cada modelo é exportado com o pré-processamento do próprio treino (salvo nos metadados):
na v3, a normalização de intensidade (também aplicada à imagem original mostrada no site)
e a máscara dos pulmões nos mapas e nos scores.
"""

from datetime import date
import json
from pathlib import Path
import shutil
from typing import Annotated

from loguru import logger
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import torch
import typer

from module_vae.config import DEVICE, PROJ_ROOT, RSNA_DIR, RunPaths
from module_vae.dataset import read_dicom
from module_vae.features import anomaly_map, anomaly_score
from module_vae.modeling.architecture import VAE
from module_vae.modeling.calibration import sigmas_for
from module_vae.modeling.predict import load_model, predict, reconstruct
from module_vae.modeling.split import DataSplits, load_splits
from module_vae.plots import HEATMAP_CMAP
from module_vae.preprocessing import Preprocessing

OUT_DIR = PROJ_ROOT / "interfaces" / "public" / "data"
ORIGINAL_SIZE = 512  # resolução do exame original mostrado no site (o modelo usa 128 px)
QUEUE_SIZE = {0: 16, 1: 4}  # exames normais (0) e com opacidade (1) na fila
GAME_SIZE = {0: 6, 1: 2}
LATENT_VALUES = np.linspace(-3, 3, 9)  # valores de cada eixo de z no gerador
INTERPOLATION_STEPS = 9
THUMB_SIZE = 40  # miniaturas do mapa do espaço latente
LABEL_NAMES = {0: "normal", 1: "opacidade"}

app = typer.Typer()


def sample_exams(
    y: np.ndarray, views: np.ndarray, n_per_class: dict[int, int], rng: np.random.Generator
) -> np.ndarray:
    """Sorteia exames de cada classe mantendo a proporção de incidências do conjunto.

    Com AP e PA, quase todas as opacidades do RSNA são AP; sortear sem olhar a incidência
    poderia esconder (ou exagerar) o atalho na fila de 20 exames.

    Args:
        y (np.ndarray): Rótulos (0 normal, 1 opacidade).
        views (np.ndarray): Incidência de cada exame.
        n_per_class (dict[int, int]): Quantos exames sortear de cada classe.
        rng (np.random.Generator): Gerador do sorteio.

    Returns:
        np.ndarray: Índices sorteados, em ordem aleatória (a ordem de chegada).
    """
    picked = []
    for label, n in n_per_class.items():
        idx = np.where(y == label)[0]
        share = pd.Series(views[idx]).value_counts(normalize=True).sort_index() * n
        counts = share.astype(int)
        # O que sobrou do arredondamento vai para as incidências com maior fração
        for view in (share - counts).sort_values(ascending=False).index[: n - counts.sum()]:
            counts[view] += 1
        for view, k in counts.items():
            picked.extend(rng.choice(idx[views[idx] == view], k, replace=False))
    picked = np.array(picked)
    rng.shuffle(picked)
    return picked


def boxes_percent(boxes: pd.DataFrame, index: int, image_size: int) -> list[list[float]]:
    """Caixas de um exame em % da imagem (x, y, largura, altura), como o CSS usa."""
    rows = boxes[boxes["exam_index"] == index]
    return [
        [round(v / image_size * 100, 2) for v in (b.x, b.y, b.width, b.height)]
        for b in rows.itertuples()
    ]


def save_gray(arr: np.ndarray, path: Path) -> None:
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(path)


def save_heat(heatmap: np.ndarray, path: Path, vmax: float) -> None:
    """Mapa de calor em RGBA, com a mesma escala de cor das figuras do pipeline."""
    rgba = HEATMAP_CMAP(np.clip(heatmap / vmax, 0, 1))
    Image.fromarray((rgba * 255).astype(np.uint8)).save(path)


class RunExport:
    """Um modelo treinado, o conjunto de teste dele e os scores pré-calculados."""

    def __init__(self, run: str, key: str):
        self.key = key  # pasta em public/data e chave no data.json ("pa", "ap_pa")
        self.paths = RunPaths(run)
        self.model, self.meta = load_model(self.paths.model, self.paths.metadata)
        self.size = self.meta["image_size"]
        self.preprocessing = Preprocessing.from_metadata(self.meta)
        self.mask = self.meta.get("mask")  # máscara dos pulmões (v3), se o modelo usa
        self.splits: DataSplits = load_splits(
            self.size,
            self.meta["views"],
            hygiene=self.meta.get("hygiene", False),
            preprocessing=self.preprocessing,
        )
        s = self.splits
        self.pred, self.rec_test, self.maps_test = predict(
            self.model, s.x_test, self.meta["sigma"], self.meta["threshold"], self.mask
        )
        self.score_test = self.pred["anomaly_score"].to_numpy()
        self.rec_val = reconstruct(self.model, s.x_val)
        score_val = anomaly_score(s.x_val, self.rec_val, self.meta["sigma"], self.mask)
        self.normal_val_scores = np.sort(score_val[s.y_val == 0])

        auc_test = roc_auc_score(s.y_test, self.score_test)
        logger.info(f"{run}: {len(s.y_test)} exames de teste, AUC {auc_test:.3f}")
        self.dir = OUT_DIR / key
        (self.dir / "exams").mkdir(parents=True, exist_ok=True)

    def url(self, name: str) -> str:
        """Endereço de um arquivo desta execução no site (public/ é servida na raiz)."""
        return f"/data/{self.key}/{name}"

    def exam(self, i: int, prefix: str = "EX-") -> dict:
        """Grava as imagens de um exame de teste e devolve o registro dele no data.json."""
        exam_id = f"{prefix}{i:04d}"
        pid = self.splits.patient_ids_test[i]
        original, _ = read_dicom(RSNA_DIR / "stage_2_train_images" / f"{pid}.dcm", ORIGINAL_SIZE)
        # O site mostra o exame como o modelo o vê (por exemplo, com o brilho normalizado)
        original = self.preprocessing.apply(original[None])[0]
        Image.fromarray(original).save(self.dir / "exams" / f"{exam_id}.jpg", quality=88)
        save_gray(self.rec_test[i, 0].numpy(), self.dir / "exams" / f"{exam_id}_rec.png")
        save_heat(
            self.maps_test[i, 0].numpy(),
            self.dir / "exams" / f"{exam_id}_heat.png",
            self.meta["vmax_map"],
        )
        score = float(self.score_test[i])
        percentile = np.searchsorted(self.normal_val_scores, score) / len(self.normal_val_scores)
        return {
            "id": exam_id,
            "label": LABEL_NAMES[int(self.splits.y_test[i])],
            "view": str(self.splits.views_test[i]),
            "score": score,
            "percentile": float(percentile),
            "flagged": bool(score >= self.meta["threshold"]),
            "boxes": boxes_percent(self.splits.boxes_test, i, self.size),
            "img": self.url(f"exams/{exam_id}.jpg"),
            "rec": self.url(f"exams/{exam_id}_rec.png"),
            "heat": self.url(f"exams/{exam_id}_heat.png"),
        }

    def model_info(self) -> dict:
        """Configuração do modelo e as métricas de teste já calculadas pelo pipeline."""
        info = {
            "run": self.paths.name,
            "views": self.meta["views"],
            "sigma": self.meta["sigma"],
            "threshold": self.meta["threshold"],
            "latent_dim": self.meta["latent_dim"],
            "auc_val": round(self.meta["auc_val"], 3),
            "n_test": len(self.splits.y_test),
            "image_size": self.size,
            "original_image_size": ORIGINAL_SIZE,
            **{
                key: self.meta[key]
                for key in ("version", "epochs", "threshold_rule", "normalize", "lung_mask")
                if key in self.meta
            },
        }
        metrics_path = self.paths.reports / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            info["test"] = {
                "roc_auc": round(metrics["roc_auc"], 3),
                "sensitivity": round(metrics["sensitivity"], 3),
                "specificity": round(metrics["specificity"], 3),
                "auc_by_view": {
                    v: round(metrics["views"][f"roc_auc_{v}"], 3) for v in self.meta["views"]
                },
                "queue_simulation": metrics["queue_simulation"],
            }
            # Com AP e PA: o quanto o score separa AP de PA entre os normais (o atalho)
            if "roc_auc_ap_vs_pa_normals" in metrics["views"]:
                info["test"]["auc_ap_vs_pa_normals"] = round(
                    metrics["views"]["roc_auc_ap_vs_pa_normals"], 3
                )
        else:
            logger.warning(f"{metrics_path} ainda não existe: o site fica sem as métricas")
        return info

    def queue(self, rng: np.random.Generator) -> tuple[list[dict], np.ndarray]:
        """A fila de trabalho: exames sorteados do teste, em ordem de chegada."""
        idx = sample_exams(self.splits.y_test, self.splits.views_test, QUEUE_SIZE, rng)
        records = [
            self.exam(i) | {"arrival": f"{8 + k * 4 // 60:02d}:{k * 4 % 60:02d}"}
            for k, i in enumerate(idx)
        ]
        return records, idx


def export_sigma(run: RunExport) -> dict:
    """Um exame com opacidade visto com cada largura σ testada na calibração.

    O exemplo é a opacidade em que o mapa do σ escolhido mais concentra energia dentro das
    caixas: é um caso favorável, escolhido para ilustrar o efeito de σ, e não um exame
    típico (a localização no conjunto todo está em docs/09-resultados-v3.md).
    """
    s, size = run.splits, run.size
    maps = run.maps_test[:, 0].numpy()
    best, best_energy = None, -1.0
    for i in np.where(s.y_test == 1)[0]:
        mask = np.zeros((size, size), bool)
        for b in s.boxes_test[s.boxes_test["exam_index"] == i].itertuples():
            mask[
                int(b.y) : int(np.ceil(b.y + b.height)), int(b.x) : int(np.ceil(b.x + b.width))
            ] = True
        energy = maps[i][mask].sum() / maps[i].sum()
        if energy > best_energy:
            best, best_energy = i, energy
    logger.info(f"Exemplo de σ: exame {best}, {best_energy:.0%} da energia do mapa nas caixas")

    steps = []
    for sigma in sigmas_for(size):
        maps_s = anomaly_map(s.x_test, run.rec_test, sigma, run.mask)
        name = f"exams/sigma_{sigma:g}.png"
        vmax = float(np.percentile(maps_s.numpy(), 99.5))
        save_heat(maps_s[best, 0].numpy(), run.dir / name, vmax)
        auc = roc_auc_score(s.y_val, anomaly_score(s.x_val, run.rec_val, sigma, run.mask))
        steps.append({"sigma": sigma, "auc_val": round(float(auc), 3), "heat": run.url(name)})

    return {"example": run.exam(best, "SG-"), "chosen": run.meta["sigma"], "steps": steps}


@torch.no_grad()
def encode_mu(model: VAE, x: torch.Tensor, batch_size: int = 256) -> np.ndarray:
    """Código μ de cada exame (o centro da distribuição q(z|x))."""
    model.eval()
    return torch.cat(
        [
            model.encode(x[i : i + batch_size].to(DEVICE))[0].cpu()
            for i in range(0, len(x), batch_size)
        ]
    ).numpy()


@torch.no_grad()
def export_latent(run: RunExport, interp_from: int, interp_to: int) -> dict:
    """Mapa do espaço latente, sprites do gerador e a interpolação entre dois exames."""
    s, model, latent_dim = run.splits, run.model, run.meta["latent_dim"]

    # Mapa: PCA dos μ, com os eixos ajustados só nos normais do treino
    pca = PCA(2).fit(encode_mu(model, s.x_train_normal))
    x_points = torch.cat([s.x_test, s.x_test_other])
    labels = [LABEL_NAMES[int(y)] for y in s.y_test] + ["outra"] * len(s.x_test_other)
    coords = pca.transform(encode_mu(model, x_points))
    cols = int(np.ceil(np.sqrt(len(x_points))))
    atlas = Image.new("L", (cols * THUMB_SIZE, cols * THUMB_SIZE))
    for k in range(len(x_points)):
        thumb = Image.fromarray((x_points[k, 0].numpy() * 255).astype(np.uint8))
        atlas.paste(
            thumb.resize((THUMB_SIZE, THUMB_SIZE)),
            ((k % cols) * THUMB_SIZE, (k // cols) * THUMB_SIZE),
        )
    atlas.save(run.dir / "latent_atlas.jpg", quality=80)
    points = [
        [round(float(c[0]), 3), round(float(c[1]), 3), lab] for c, lab in zip(coords, labels)
    ]

    # Gerador: cada eixo de z de −3 a +3, com os outros em zero (um sprite por eixo)
    for d in range(latent_dim):
        z = torch.zeros(len(LATENT_VALUES), latent_dim)
        z[:, d] = torch.tensor(LATENT_VALUES, dtype=torch.float32)
        frames = model.decode(z.to(DEVICE)).cpu()[:, 0].numpy()
        save_gray(np.concatenate(list(frames), axis=1), run.dir / f"latent_dim{d}.png")

    # Interpolação em linha reta entre os μ de dois exames
    mu = torch.tensor(encode_mu(model, s.x_test[[interp_from, interp_to]]))
    t = torch.linspace(0, 1, INTERPOLATION_STEPS).unsqueeze(1)
    frames = model.decode(((1 - t) * mu[0] + t * mu[1]).to(DEVICE)).cpu()[:, 0].numpy()
    save_gray(np.concatenate(list(frames), axis=1), run.dir / "interpolation.png")

    return {
        "values": LATENT_VALUES.tolist(),
        "points": points,
        "atlas": run.url("latent_atlas.jpg"),
        "atlas_cols": cols,
        "dims": [run.url(f"latent_dim{d}.png") for d in range(latent_dim)],
        "interpolation": {
            "sprite": run.url("interpolation.png"),
            "steps": INTERPOLATION_STEPS,
            "from": run.exam(interp_from),
            "to": run.exam(interp_to),
        },
    }


@app.command()
def main(
    run: Annotated[str, typer.Option(help="Modelo principal (só PA): alimenta as três abas.")] = (
        "rsna_pa_v3"
    ),
    shortcut_run: Annotated[
        str, typer.Option(help="Modelo com AP e PA para o botão da fila ('' para não exportar).")
    ] = "rsna_ap_pa_v3",
    seed: Annotated[int, typer.Option(help="Semente do sorteio dos exames.")] = 7,
):
    # A pasta é toda gerada aqui: apagar evita sobrar imagem de uma exportação anterior.
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    rng = np.random.default_rng(seed)

    main_run = RunExport(run, "pa")
    queue, queue_idx = main_run.queue(rng)

    # O jogo usa exames que não estão na fila
    s = main_run.splits
    rest = np.setdiff1d(np.arange(len(s.y_test)), queue_idx)
    game_idx = rest[sample_exams(s.y_test[rest], s.views_test[rest], GAME_SIZE, rng)]
    game = [main_run.exam(i, "JG-") for i in game_idx]

    # Interpolação do primeiro normal da fila até a primeira opacidade
    interp_from = next(i for i in queue_idx if s.y_test[i] == 0)
    interp_to = next(i for i in queue_idx if s.y_test[i] == 1)

    data = {
        "generated": date.today().isoformat(),
        "main": "pa",
        "runs": {"pa": {"model": main_run.model_info(), "queue": queue}},
        "game": game,
        "sigma": export_sigma(main_run),
        "latent": export_latent(main_run, interp_from, interp_to),
    }

    if shortcut_run:
        if RunPaths(shortcut_run).model.exists():
            other = RunExport(shortcut_run, "ap_pa")
            other_queue, _ = other.queue(np.random.default_rng(seed))
            data["runs"]["ap_pa"] = {"model": other.model_info(), "queue": other_queue}
        else:
            logger.warning(f"{shortcut_run} não tem modelo salvo: o botão AP + PA fica desligado")

    (OUT_DIR / "data.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    for key, block in data["runs"].items():
        q = block["queue"]
        n_flag = sum(e["flagged"] for e in q)
        logger.info(f"Fila {key}: {len(q)} exames, {n_flag} acima do limiar")
    size_mb = sum(f.stat().st_size for f in OUT_DIR.rglob("*") if f.is_file()) / 1e6
    logger.success(f"Dados da demo em {OUT_DIR} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    app()
