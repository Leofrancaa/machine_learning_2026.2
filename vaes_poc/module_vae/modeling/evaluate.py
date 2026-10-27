"""Avaliação final no conjunto de teste e simulação da fila de triagem (Seção 8).

Além das métricas de detecção, as caixas e as classes do RSNA permitem três avaliações:

- outras anomalias: o VAE sinaliza exames anormais que não são pneumonia?
- localização: o mapa de calor aponta a lesão marcada pelo radiologista?
- incidência: o score mede a doença ou só a diferença entre exames AP e PA?

A partir da v2, também o topo da lista ordenada (precision@k e recall@k) e a AUC por
equipamento (PixelSpacing). Localização, topo da lista e equipamento são calculados na
validação e no teste: as métricas de validação servem para comparar configurações sem
olhar o teste.
"""

from loguru import logger
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
import torch


def metrics_at_threshold(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    """Sensibilidade, especificidade, precisão, F1, acurácia e matriz de confusão em um limiar.

    Com threshold = inf, ninguém passa do limiar: é o "modelo" que diz "normal" para
    todos, a referência que mostra por que a acurácia engana com classes desbalanceadas.
    """
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    return {
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "precision": tp / max(tp + fp, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "accuracy": (tp + tn) / len(y),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def evaluate_model(y_test: np.ndarray, score_test: np.ndarray, threshold: float) -> dict:
    """Avaliação final do score no conjunto de teste.

    - AUC-ROC: separação geral, independente de limiar.
    - AP (precisão média): mais informativa com classes desbalanceadas. A AP de um score
      aleatório é a proporção de positivos no teste, não 0,5.
    - Sensibilidade e especificidade no limiar calibrado na validação.
    """
    results = {
        "roc_auc": float(roc_auc_score(y_test, score_test)),
        "average_precision": float(average_precision_score(y_test, score_test)),
        "threshold": threshold,
        **metrics_at_threshold(y_test, score_test, threshold),
        "accuracy_all_normal": float((y_test == 0).mean()),
    }
    for key in ["sensitivity", "specificity", "precision", "f1", "accuracy"]:
        results[key] = float(results[key])

    logger.info("=" * 60)
    logger.success("AVALIAÇÃO NO TESTE")
    logger.info(f"AUC-ROC: {results['roc_auc']:.3f}")
    logger.info(f"AP: {results['average_precision']:.3f} (acaso: {y_test.mean():.3f})")
    logger.info(f"Sensibilidade: {results['sensitivity']:.3f}")
    logger.info(f"Especificidade: {results['specificity']:.3f}")
    logger.info(f"Precisão (VPP): {results['precision']:.3f}")
    logger.info(f"F1: {results['f1']:.3f}")
    logger.info(
        f"Acurácia: {results['accuracy']:.3f} "
        f'("normal" para todos: {results["accuracy_all_normal"]:.3f})'
    )
    logger.info(f"Matriz de confusão: {results['confusion_matrix']}")

    return results


def compare_thresholds(
    thresholds: dict[str, float],
    y_val: np.ndarray,
    score_val: np.ndarray,
    y_test: np.ndarray,
    score_test: np.ndarray,
) -> pd.DataFrame:
    """Compara as métricas de vários limiares na validação e no teste.

    Se a especificidade do teste ficar bem abaixo da validação, o "normal" mudou entre os
    conjuntos e o limiar precisaria ser recalibrado. A última
    linha é a referência "normal para todos" (limiar infinito).
    """
    columns = ["sensitivity", "specificity", "precision", "f1", "accuracy"]
    rows = []
    for name, threshold in {**thresholds, "normal para todos (referência)": np.inf}.items():
        for split, y, score in [("val", y_val, score_val), ("test", y_test, score_test)]:
            m = metrics_at_threshold(y, score, threshold)
            rows.append({"threshold": name, "split": split, **{c: m[c] for c in columns}})

    table = pd.DataFrame(rows).set_index(["threshold", "split"])
    logger.info(f"Limiares de decisão:\n{table.round(3).to_string()}")

    return table


def ranking_at_k(
    y: np.ndarray, score: np.ndarray, ks: tuple[int, ...] = (10, 25, 50, 100)
) -> dict:
    """Precision@k e recall@k: o que há entre os k exames com maior score?

    - precision@k: fração de positivos entre os k primeiros (acaso: a proporção de
      positivos no conjunto, guardada em "prevalence").
    - recall@k: fração de todos os positivos do conjunto que estão entre os k primeiros.

    Diferente da fila simulada, usa o conjunto inteiro como uma única lista, com a
    prevalência do próprio conjunto.
    """
    top = y[np.argsort(-score, kind="stable")]
    results = {"prevalence": float(y.mean())}
    for k in ks:
        results[f"precision_at_{k}"] = float(top[:k].mean())
        results[f"recall_at_{k}"] = float(top[:k].sum() / max(y.sum(), 1))

    logger.info(
        "Topo da lista: "
        + ", ".join(f"P@{k} {results[f'precision_at_{k}']:.2f}" for k in ks)
        + f" (acaso: {results['prevalence']:.2f})"
    )
    return results


def evaluate_equipment(
    y: np.ndarray, score: np.ndarray, equipment: np.ndarray, min_positives: int = 10
) -> dict:
    """AUC dentro de cada equipamento (PixelSpacing).

    Equipamento, brilho e incidência andam juntos no RSNA (docs/06-eda-rsna.md, pergunta
    5). Se a AUC variar muito entre equipamentos, parte do score mede a aquisição. Grupos
    com menos de min_positives positivos (ou sem normais) não têm AUC confiável e ficam só
    com as contagens.

    Returns:
        dict: {equipamento: {"n", "n_positive", "roc_auc"}}.
    """
    results = {}
    for group in np.unique(equipment):
        mask = equipment == group
        n_positive = int(y[mask].sum())
        results[group] = {"n": int(mask.sum()), "n_positive": n_positive}
        if n_positive >= min_positives and n_positive < mask.sum():
            results[group]["roc_auc"] = float(roc_auc_score(y[mask], score[mask]))

    table = pd.DataFrame(results).T
    logger.info(f"AUC por equipamento (PixelSpacing):\n{table.to_string()}")
    return results


def simulate_queue(
    y: np.ndarray,
    score: np.ndarray,
    n_queues: int = 2000,
    queue_size: int = 100,
    prevalence: float = 0.10,
    seed: int = 0,
) -> dict:
    """Simula filas realistas e compara a ordem de chegada com a ordem pelo score.

    A proporção de positivos no teste não é a de uma fila real. Aqui sorteamos filas de
    queue_size exames com a prevalência dada e medimos onde os positivos (pneumonias ou
    opacidades) ficam em cada ordenação.

    Returns:
        dict: Posição média de um positivo na fila (1 a queue_size) e quantos positivos
            aparecem entre os 10 primeiros, nas duas ordens.
    """
    rng = np.random.default_rng(seed)
    normal_idx, pneumonia_idx = np.where(y == 0)[0], np.where(y == 1)[0]
    n_pneumonia = round(queue_size * prevalence)

    pos_arrival, pos_model, top10_arrival, top10_model = [], [], [], []
    for _ in range(n_queues):
        queue = np.concatenate(
            [
                rng.choice(normal_idx, queue_size - n_pneumonia, replace=False),
                rng.choice(pneumonia_idx, n_pneumonia, replace=False),
            ]
        )
        rng.shuffle(queue)  # ordem de chegada aleatória
        queue_model = queue[np.argsort(-score[queue])]  # ordem pelo score do VAE

        pos_arrival.append((np.where(y[queue] == 1)[0] + 1).mean())
        pos_model.append((np.where(y[queue_model] == 1)[0] + 1).mean())
        top10_arrival.append(y[queue[:10]].sum())
        top10_model.append(y[queue_model[:10]].sum())

    results = {
        "mean_position_arrival": float(np.mean(pos_arrival)),
        "mean_position_model": float(np.mean(pos_model)),
        "top10_arrival": float(np.mean(top10_arrival)),
        "top10_model": float(np.mean(top10_model)),
    }

    logger.success(f"FILA SIMULADA ({queue_size} exames, {prevalence:.0%} de positivos)")
    logger.info(
        "Posição média de um positivo: "
        f"{results['mean_position_arrival']:.0f} → {results['mean_position_model']:.0f}"
    )
    logger.info(
        f"Positivos entre os 10 primeiros: {results['top10_arrival']:.1f} → "
        f"{results['top10_model']:.1f}"
    )

    return results


def evaluate_other_anomalies(
    score_normal: np.ndarray, score_other: np.ndarray, threshold: float
) -> dict:
    """O VAE sinaliza exames anormais que não são pneumonia ("anormal sem opacidade")?

    σ e limiar foram escolhidos com opacidades; estes exames não influenciaram nenhuma
    escolha. É o teste da promessa central da detecção de anomalias: sinalizar alterações
    que o modelo nunca viu e para as quais não foi ajustado.
    """
    y = np.r_[np.zeros(len(score_normal)), np.ones(len(score_other))]
    results = {
        "roc_auc_other": float(roc_auc_score(y, np.r_[score_normal, score_other])),
        "sensitivity_other": float((score_other >= threshold).mean()),
        "n_other": len(score_other),
    }

    logger.success("OUTRAS ANOMALIAS (anormal sem opacidade × normal)")
    logger.info(f"AUC-ROC: {results['roc_auc_other']:.3f}")
    logger.info(f"Sinalizados no limiar: {results['sensitivity_other']:.1%}")

    return results


def evaluate_localization(
    maps: torch.Tensor,
    boxes: pd.DataFrame,
    split: str = "teste",
    lung_mask: torch.Tensor | None = None,
) -> dict:
    """O mapa de calor aponta a lesão? Usa as caixas marcadas pelos radiologistas.

    - pointing game: o pixel mais quente do mapa cai dentro de alguma caixa?
    - energia nas caixas: que fração do "calor" total do mapa fica dentro das caixas?

    Um mapa sem informação (uniforme ou sorteado) acerta, em média, a fração da imagem
    coberta pelas caixas: essa é a referência do acaso para as duas métricas.

    Também mede se o score acompanha a extensão da lesão (área das caixas), uma
    aproximação grosseira de gravidade que o dataset permite testar.

    Args:
        maps (torch.Tensor): Mapas de calor de um conjunto (N, 1, H, W).
        boxes (pd.DataFrame): Caixas (exam_index, x, y, width, height) em pixels.
        split (str): Nome do conjunto, só para o log.
        lung_mask (torch.Tensor | None): Máscara dos pulmões (ablação C1). Com ela, o
            mapa só existe dentro dos pulmões, e um mapa sem informação acertaria a
            fração da máscara coberta pelas caixas, maior que a da imagem inteira. Essa
            referência do acaso vai em box_fraction_in_mask.
    """
    size = maps.shape[-1]
    lung = None if lung_mask is None else lung_mask.numpy().astype(bool)
    hits, energy, area, area_in_lung, score = [], [], [], [], []
    for i, group in boxes.groupby("exam_index"):
        mask = np.zeros((size, size), dtype=bool)
        for box in group.itertuples():
            x0, y0 = max(int(box.x), 0), max(int(box.y), 0)
            x1 = min(int(np.ceil(box.x + box.width)), size)
            y1 = min(int(np.ceil(box.y + box.height)), size)
            mask[y0:y1, x0:x1] = True

        heatmap = maps[i, 0].numpy()
        hits.append(mask.flat[heatmap.argmax()])
        energy.append(heatmap[mask].sum() / heatmap.sum())
        area.append(mask.mean())
        if lung is not None:
            area_in_lung.append((mask & lung).sum() / lung.sum())
        score.append(heatmap.mean())

    results = {
        "n_exams": len(hits),
        "pointing_game": float(np.mean(hits)),
        "energy_in_boxes": float(np.mean(energy)),
        "box_area_fraction": float(np.mean(area)),
        "extent_spearman": float(spearmanr(area, score)[0]),
    }
    if lung is not None:
        results["box_fraction_in_mask"] = float(np.mean(area_in_lung))

    logger.success(f"LOCALIZAÇÃO, {split} ({results['n_exams']} exames com caixas)")
    logger.info(
        f"Pixel mais quente dentro da caixa: {results['pointing_game']:.1%} "
        f"(acaso: {results['box_area_fraction']:.1%})"
    )
    logger.info(
        f"Fração do calor dentro das caixas: {results['energy_in_boxes']:.1%} "
        f"(acaso: {results['box_area_fraction']:.1%})"
    )
    if lung is not None:
        logger.info(f"Acaso dentro da máscara dos pulmões: {results['box_fraction_in_mask']:.1%}")
    logger.info(f"Correlação score × área da lesão (Spearman): {results['extent_spearman']:.2f}")

    return results


def evaluate_views(y: np.ndarray, score: np.ndarray, views: np.ndarray) -> dict:
    """Separa o efeito da incidência (AP × PA) do efeito da doença.

    - AUC dentro de cada incidência: compara só exames da mesma incidência.
    - AUC "AP × PA" entre normais: se for alta, o score mede a incidência, não a doença.
    """
    results = {}
    for view in np.unique(views):
        mask = views == view
        if len(np.unique(y[mask])) == 2:
            results[f"roc_auc_{view}"] = float(roc_auc_score(y[mask], score[mask]))

    normal = y == 0
    if len(np.unique(views[normal])) == 2:
        results["roc_auc_ap_vs_pa_normals"] = float(
            roc_auc_score(views[normal] == "AP", score[normal])
        )

    logger.success("INCIDÊNCIA")
    for name, value in results.items():
        logger.info(f"{name}: {value:.3f}")

    return results
