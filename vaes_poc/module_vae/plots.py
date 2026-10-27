"""Figuras do pipeline, todas com o mesmo estilo. Cada execução salva as suas em
reports/<execução>/figures."""

from pathlib import Path

from loguru import logger
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve
import torch
from torchvision.utils import make_grid

# Cores fixas por significado em todas as figuras
NORMAL_COLOR = "#2a78d6"  # azul    → exames normais (e o VAE nas comparações)
POSITIVE_COLOR = "#eb6834"  # laranja → exames com opacidade
NEUTRAL_COLOR = "#a3a29b"  # cinza   → referências e baselines
TEXT_COLOR_2 = "#52514e"
RULE_COLOR = "#c3c2b7"

# Mapa de calor sobreposto ao raio-X: transparente → laranja → vermelho
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "calor", [(0.92, 0.41, 0.20, 0.0), (0.92, 0.41, 0.20, 0.55), (0.89, 0.29, 0.28, 0.95)]
)


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
            "savefig.facecolor": "#fcfcfb",
            "axes.edgecolor": RULE_COLOR,
            "axes.labelcolor": TEXT_COLOR_2,
            "text.color": "#0b0b0b",
            "xtick.color": TEXT_COLOR_2,
            "ytick.color": TEXT_COLOR_2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e8e7e3",
            "grid.linewidth": 0.6,
            "lines.linewidth": 2,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "legend.frameon": False,
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Figura salva em {path}")


def show_grid(images: torch.Tensor, n_cols: int, ax: plt.Axes, title: str | None = None) -> None:
    """Monta um mosaico com várias imagens (N, 1, H, W) e exibe em tons de cinza."""
    grid = make_grid(images, nrow=n_cols, padding=2, pad_value=1.0)[0]
    ax.imshow(grid, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    if title:
        ax.set_title(title)


def plot_training_curves(
    history: dict[str, list[float]],
    kl_warmup_epochs: int,
    path: Path,
) -> None:
    epochs = np.arange(1, len(history["rec_train"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))

    axes[0].plot(epochs, history["rec_train"], color=NORMAL_COLOR, label="treino")
    axes[0].plot(
        epochs,
        history["rec_val"],
        color=NEUTRAL_COLOR,
        linestyle="--",
        label="validação (normais)",
    )
    axes[0].set_title("Erro de reconstrução do VAE (BCE por imagem)")
    axes[0].set_xlabel("época")
    axes[0].legend()

    axes[1].plot(epochs, history["kl_train"], color=NORMAL_COLOR)
    axes[1].axvspan(1, min(kl_warmup_epochs, len(epochs)), color="#e8e7e3", alpha=0.6, lw=0)
    axes[1].set_title("Divergência KL por imagem (faixa cinza = KL warm-up)")
    axes[1].set_xlabel("época")

    save_figure(fig, path)


def plot_lung_mask(mean_image: np.ndarray, mask: np.ndarray, path: Path) -> None:
    """Máscara dos pulmões (ablação C1) sobre a imagem média dos normais de treino."""
    fig, ax = plt.subplots(figsize=(3.6, 3.8))
    ax.imshow(mean_image, cmap="gray", vmin=0, vmax=1)
    ax.contour(mask, levels=[0.5], colors=[POSITIVE_COLOR], linewidths=1.5)
    ax.set_title(f"Máscara dos pulmões ({mask.mean():.0%} da imagem)", fontsize=10)
    ax.axis("off")
    save_figure(fig, path)


def plot_generated_samples(samples: torch.Tensor, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 2.2))
    show_grid(samples, 8, ax, "Raios-X sintéticos: decoder aplicado a z ~ N(0, I)")
    save_figure(fig, path)


def plot_sigma_selection(
    auc_val: dict[float, float],
    auc_test: dict[float, float],
    chosen_sigma: float,
    path: Path,
) -> None:
    sigmas = list(auc_val)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(
        sigmas,
        list(auc_val.values()),
        marker="o",
        markersize=5,
        color=NORMAL_COLOR,
        label="validação",
    )
    ax.plot(
        sigmas,
        list(auc_test.values()),
        marker="o",
        markersize=5,
        color=NEUTRAL_COLOR,
        linestyle="--",
        label="teste",
    )
    ax.axhline(0.5, color=RULE_COLOR, lw=1)
    ax.axvline(chosen_sigma, color=RULE_COLOR, lw=1, linestyle=":")
    ax.text(
        chosen_sigma + 0.1, 0.56, f"σ escolhido = {chosen_sigma}", color=TEXT_COLOR_2, fontsize=9
    )
    ax.set_xlabel("σ da suavização do resíduo (pixels)")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Suavizar o resíduo antes de medir o erro melhora o score")
    ax.legend(loc="lower right")
    save_figure(fig, path)


def plot_test_evaluation(
    y: np.ndarray, score: np.ndarray, path: Path, positive_name: str = "opacidade"
) -> None:
    """Curva ROC, curva precisão × revocação e distribuição do score no teste."""
    fpr, tpr, _ = roc_curve(y, score)
    precision, recall, _ = precision_recall_curve(y, score)

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
    axes[0].plot(fpr, tpr, color=NORMAL_COLOR)
    axes[0].plot([0, 1], [0, 1], color=RULE_COLOR, lw=1, linestyle="--")
    axes[0].set_title("Curva ROC")
    axes[0].set_xlabel("taxa de falsos positivos (1 − especificidade)")
    axes[0].set_ylabel("sensibilidade")

    axes[1].plot(recall, precision, color=NORMAL_COLOR)
    axes[1].axhline(y.mean(), color=RULE_COLOR, lw=1, linestyle="--")
    axes[1].set_title("Precisão × revocação (tracejado = acaso)")
    axes[1].set_xlabel("revocação (sensibilidade)")
    axes[1].set_ylabel("precisão")

    bins = np.linspace(0, np.percentile(score, 99), 40)
    for label, color, name in [(0, NORMAL_COLOR, "normal"), (1, POSITIVE_COLOR, positive_name)]:
        axes[2].hist(
            score[y == label], bins=bins, color=color, alpha=0.55, density=True, label=name
        )
    axes[2].set_title("Distribuição do score de anomalia")
    axes[2].set_xlabel("score")
    axes[2].set_ylabel("densidade")
    axes[2].legend()

    save_figure(fig, path)


def plot_heatmaps(
    x: torch.Tensor,
    x_rec: torch.Tensor,
    maps: torch.Tensor,
    scores: np.ndarray,
    indices: list[int],
    vmax: float,
    title: str,
    path: Path,
    labels: np.ndarray | None = None,
    boxes: pd.DataFrame | None = None,
) -> None:
    """Original, reconstrução e mapa de calor de alguns exames.

    A escala de cor (vmax) é a mesma para todos os exames: normalizar por exame faria
    qualquer exame parecer "quente".

    Args:
        labels (np.ndarray | None): Nome da classe de cada exame ("normal", "opacidade"...).
        boxes (pd.DataFrame | None): Caixas das lesões (exam_index, x, y, width, height),
            desenhadas em azul sobre o mapa de calor.
    """
    fig, axes = plt.subplots(3, len(indices), figsize=(2.1 * len(indices), 6.6), squeeze=False)
    for c, i in enumerate(indices):
        header = f"score {scores[i] * 1e3:.2f}"
        if labels is not None:
            header = f"{labels[i]}\n{header}"
        axes[0, c].imshow(x[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[0, c].set_title(header, fontsize=9, fontweight="normal")
        axes[1, c].imshow(x_rec[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[2, c].imshow(x[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[2, c].imshow(maps[i, 0], cmap=HEATMAP_CMAP, vmin=0, vmax=vmax)
        if boxes is not None:
            for box in boxes[boxes["exam_index"] == i].itertuples():
                axes[2, c].add_patch(
                    Rectangle(
                        (box.x, box.y),
                        box.width,
                        box.height,
                        fill=False,
                        edgecolor=NORMAL_COLOR,
                        lw=1.2,
                    )
                )
        for r in range(3):
            axes[r, c].axis("off")

    for r, name in enumerate(["original", "reconstrução", "mapa de calor"]):
        axes[r, 0].text(
            -0.15, 0.5, name, transform=axes[r, 0].transAxes, rotation=90, va="center", ha="right"
        )
    fig.suptitle(f"{title} (score × 1000)", fontweight="bold")
    save_figure(fig, path)


def plot_method_comparison(table: pd.DataFrame, path: Path) -> None:
    """Barras de AUC no teste; hachurado = usa rótulos da doença."""
    names = list(table.index)[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for k, name in enumerate(names):
        row = table.loc[name]
        if name == "VAE":
            color, hatch = NORMAL_COLOR, None
        elif row["uses_labels"]:
            color, hatch = "#e8e7e3", "///"
        else:
            color, hatch = NEUTRAL_COLOR, None
        ax.barh(
            k,
            row["auc_test"],
            xerr=row["auc_test_std"] or None,
            color=color,
            hatch=hatch,
            edgecolor=RULE_COLOR if hatch else color,
            height=0.62,
            error_kw={"ecolor": TEXT_COLOR_2, "capsize": 3, "lw": 1},
        )
        ax.text(
            row["auc_test"] + 0.01 + row["auc_test_std"],
            k,
            f"{row['auc_test']:.2f}",
            va="center",
            fontsize=9,
            color=TEXT_COLOR_2,
        )

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.axvline(0.5, color=RULE_COLOR, lw=1, linestyle="--")
    ax.set_xlim(0.4, 1.0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("AUC-ROC no conjunto de teste")
    ax.set_title("Comparação de métodos (hachurado = usa rótulos da doença)")
    save_figure(fig, path)


def plot_gain_curve(
    y: np.ndarray, score: np.ndarray, path: Path, positive_name: str = "opacidades"
) -> None:
    """Revisando os primeiros X% da fila ordenada pelo score, que fração dos exames
    alterados é encontrada?"""
    order = np.argsort(-score)
    reviewed = np.arange(1, len(order) + 1) / len(order)
    found = np.cumsum(y[order]) / y.sum()
    ideal = np.minimum(np.arange(1, len(order) + 1) / y.sum(), 1)

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.plot(reviewed, found, color=NORMAL_COLOR, label="ordenada pelo VAE")
    ax.plot(reviewed, ideal, color=NEUTRAL_COLOR, linestyle="--", label="ordem ideal")
    ax.plot([0, 1], [0, 1], color=NEUTRAL_COLOR, linestyle=":", label="ordem de chegada")
    k = int(0.3 * len(order))
    ax.annotate(
        f"revisando 30% da fila,\nencontra {found[k - 1]:.0%} das {positive_name}",
        xy=(0.3, found[k - 1]),
        xytext=(0.36, 0.2),
        fontsize=9,
        color=NORMAL_COLOR,
        arrowprops={"arrowstyle": "-", "color": NORMAL_COLOR, "lw": 1},
    )
    ax.set_xlabel("fração da fila revisada")
    ax.set_ylabel(f"fração das {positive_name} encontradas")
    ax.set_title("Curva de ganho da fila ordenada pelo VAE (teste)")
    ax.legend(loc="lower right")
    save_figure(fig, path)


def plot_ablation(latent: pd.Series, beta: pd.Series, path: Path) -> None:
    """AUC de validação em função do tamanho do latente e de β."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, results, name in [
        (axes[0], latent, "dimensão do espaço latente"),
        (axes[1], beta, "β (peso do KL)"),
    ]:
        ax.plot(range(len(results)), results.values, marker="o", markersize=6, color=NORMAL_COLOR)
        ax.set_xticks(range(len(results)))
        ax.set_xticklabels([str(v) for v in results.index])
        ax.set_xlabel(name)
        ax.set_title(f"AUC (validação) × {name}")
    axes[0].set_ylabel("AUC-ROC (validação)")
    save_figure(fig, path)


def plot_ablation_rule(
    seeds: dict[str, pd.DataFrame],
    min_auc_gain: float,
    min_lift_gain: float,
    accepted: dict[str, bool | None],
    path: Path,
) -> None:
    """Cada configuração das ablações: as sementes (pontos) e a média ± desvio-padrão.

    Dois painéis, cada um com a sua escala: AUC de validação e localização acima do acaso.
    A linha cinza é a média da v2; a tracejada, o ganho mínimo exigido pela regra.

    Args:
        seeds (dict[str, pd.DataFrame]): Por configuração (a primeira é a referência), as
            colunas "auc" e "lift" de cada semente.
        min_auc_gain (float): Ganho mínimo de AUC da regra.
        min_lift_gain (float): Ganho mínimo de localização acima do acaso.
        accepted (dict[str, bool | None]): Decisão da regra por configuração.
        path (Path): Onde salvar.
    """
    names = list(seeds)
    reference = seeds[names[0]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.55 * len(names) + 1.6), sharey=True)
    for ax, column, gain, title, scale in [
        (axes[0], "auc", min_auc_gain, "AUC (validação)", 1),
        (axes[1], "lift", min_lift_gain, "Pixel mais quente na caixa, acima do acaso (p.p.)", 100),
    ]:
        base = reference[column].mean() * scale
        ax.axvline(base, color=NEUTRAL_COLOR, lw=1)
        ax.axvline(base + gain * scale, color=NEUTRAL_COLOR, lw=1, ls="--")
        for row, name in enumerate(names):
            values = seeds[name][column].to_numpy() * scale
            y = len(names) - 1 - row
            color = NORMAL_COLOR if accepted.get(name) is not False else NEUTRAL_COLOR
            ax.scatter(values, np.full(len(values), y), s=18, color=color, alpha=0.45, lw=0)
            ax.errorbar(
                values.mean(), y, xerr=values.std(ddof=1), fmt="o", ms=8, color=color, capsize=0
            )
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    labels = {True: "  (aceita)", False: "  (rejeitada)", None: ""}
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels([n + labels[accepted.get(n)] for n in reversed(names)])
    fig.suptitle(
        "Ablações × v2: sementes (pontos claros) e média ± desvio-padrão. "
        "Linha cheia: média da v2; tracejada: ganho mínimo da regra",
        fontsize=9,
        color=TEXT_COLOR_2,
    )
    save_figure(fig, path)
