"""Pipeline completo: dados → treino → calibração → avaliação → comparação.

uv run python -m module_vae.main                  # só incidência PA (resultado principal)
uv run python -m module_vae.main --views AP,PA    # AP e PA juntas: mostra o atalho AP/PA
uv run python -m module_vae.main --epochs 2 --n-seeds 1 --no-baselines   # teste rápido

Cada execução é registrada no MLflow (parâmetros, curvas de treino, métricas e artefatos);
use --label para dar um nome a ela, por exemplo --label v2-linha-de-base.

Os resultados vão para pastas da versão (--version, padrão em config.VERSION): por exemplo
models/rsna_pa_v2 e reports/rsna_pa_v2. O pipeline não grava numa pasta que já existe,
a não ser com --overwrite; as pastas da v1 (rsna_pa, rsna_ap_pa) ficam protegidas.
"""

import json
from typing import Annotated

from loguru import logger
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import typer

from module_vae.config import (
    BATCH_SIZE,
    BETA,
    DEVICE,
    EPOCHS,
    IMAGE_SIZE,
    KL_WARMUP_EPOCHS,
    LATENT_DIM,
    LEARNING_RATE,
    MAX_VALID_AGE,
    MIN_AGE,
    SEEDS,
    THRESHOLD_PERCENTILE,
    VERSION,
    RunPaths,
    run_name,
)
from module_vae.features import anomaly_map, anomaly_score
from module_vae.modeling.architecture import VAE, describe_architecture
from module_vae.modeling.baselines import compare_methods
from module_vae.modeling.calibration import calibrate_threshold, select_sigma, youden_threshold
from module_vae.modeling.evaluate import (
    compare_thresholds,
    evaluate_equipment,
    evaluate_localization,
    evaluate_model,
    evaluate_other_anomalies,
    evaluate_views,
    ranking_at_k,
    simulate_queue,
)
from module_vae.modeling.predict import predict, reconstruct
from module_vae.modeling.split import load_splits
from module_vae.modeling.train import save_model, set_seed, train_autoencoder
from module_vae.plots import (
    apply_style,
    plot_gain_curve,
    plot_generated_samples,
    plot_heatmaps,
    plot_lung_mask,
    plot_method_comparison,
    plot_sigma_selection,
    plot_test_evaluation,
    plot_training_curves,
)
from module_vae.preprocessing import Preprocessing
from module_vae.preprocessing import lung_mask as build_lung_mask
from module_vae.tracking import flatten, log_comparison, log_history, tracked_run

app = typer.Typer()


@app.command()
def main(
    views: Annotated[str, typer.Option(help="Incidências: PA ou AP,PA.")] = "PA",
    image_size: Annotated[int, typer.Option(help="Resolução das imagens.")] = IMAGE_SIZE,
    latent_dim: Annotated[int, typer.Option(help="Tamanho do código latente z.")] = LATENT_DIM,
    epochs: Annotated[int, typer.Option(help="Épocas de treino de cada autoencoder.")] = EPOCHS,
    n_seeds: Annotated[int, typer.Option(help="Sementes do VAE e do autoencoder comum.")] = len(
        SEEDS
    ),
    baselines: Annotated[
        bool, typer.Option(help="Compara com PCA, autoencoder comum e CNN.")
    ] = True,
    label: Annotated[
        str, typer.Option(help="Nome da execução no MLflow, por exemplo v2-linha-de-base.")
    ] = "manual",
    version: Annotated[
        str, typer.Option(help="Versão dos resultados: define as pastas (rsna_pa_v2...).")
    ] = VERSION,
    hygiene: Annotated[
        bool, typer.Option(help="Tira duplicata, idades inválidas e menores de 18.")
    ] = True,
    overwrite: Annotated[
        bool, typer.Option(help="Permite gravar por cima de uma versão que já existe.")
    ] = False,
    seed: Annotated[
        int | None,
        typer.Option(help="Semente do VAE principal (o dos mapas e do modelo salvo)."),
    ] = None,
    crop_borders: Annotated[
        bool, typer.Option(help="Ablação A: recorta as molduras pretas.")
    ] = False,
    normalize: Annotated[
        bool, typer.Option(help="Ablação B: estica o brilho de cada imagem (p1–p99).")
    ] = False,
    lung_mask: Annotated[
        bool, typer.Option(help="Ablação C1: score só dentro da máscara média dos pulmões.")
    ] = False,
):
    view_list = sorted(v.strip().upper() for v in views.split(","))
    paths = RunPaths(run_name(view_list, version))
    if paths.exists and not overwrite:
        logger.error(
            f"A versão {version} já tem resultados em {paths.reports}. Use outra --version "
            "ou --overwrite (as pastas da v1 nunca devem ser sobrescritas)."
        )
        raise typer.Exit(1)
    # A primeira semente é a do VAE principal; as outras só entram na comparação de métodos.
    # Com --seed, ela vem primeiro, para medir a variação entre treinos (mapas, localização).
    preprocessing = Preprocessing(crop_borders, normalize, lung_mask)
    seeds = SEEDS[:n_seeds] if seed is None else [seed, *[s for s in SEEDS if s != seed]][:n_seeds]
    params = {
        "version": version,
        "hygiene": hygiene,
        "min_age": MIN_AGE if hygiene else None,
        "max_valid_age": MAX_VALID_AGE if hygiene else None,
        "views": ",".join(view_list),
        "image_size": image_size,
        "latent_dim": latent_dim,
        "beta": BETA,
        "epochs": epochs,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "kl_warmup_epochs": KL_WARMUP_EPOCHS,
        "seeds": ",".join(str(s) for s in seeds),
        "baselines": baselines,
        "threshold_percentile": THRESHOLD_PERCENTILE,
        **preprocessing.as_dict(),
    }
    # O experimento no MLflow reúne todas as versões (rsna_pa ou rsna_ap_pa), para
    # compará-las lado a lado; cada versão tem as próprias pastas.
    with tracked_run(run_name(view_list), label, params, tags={"version": version}):
        run_pipeline(
            paths,
            view_list,
            image_size,
            latent_dim,
            epochs,
            seeds,
            baselines,
            hygiene,
            version,
            preprocessing,
        )


def run_pipeline(
    paths: RunPaths,
    view_list: list[str],
    image_size: int,
    latent_dim: int,
    epochs: int,
    seeds: list[int],
    baselines: bool,
    hygiene: bool = True,
    version: str = VERSION,
    preprocessing: Preprocessing | None = None,
) -> None:
    """Executa o pipeline dentro de uma execução já aberta no MLflow."""
    preprocessing = preprocessing or Preprocessing()
    positive = "opacidade"

    logger.info(f"Execução: {paths.name} | {image_size} px | dispositivo: {DEVICE}")
    apply_style()

    # =============================================
    # DADOS
    # =============================================

    splits = load_splits(image_size, view_list, hygiene=hygiene, preprocessing=preprocessing)
    y_test = splits.y_test

    # Ablação C1: a máscara sai só dos normais do treino
    mask = None
    if preprocessing.lung_mask:
        mean_image = splits.x_train_normal.mean(0)[0].numpy()
        mask = torch.from_numpy(build_lung_mask(mean_image))
        plot_lung_mask(mean_image, mask.numpy(), paths.figures / "lung_mask.png")
        mlflow.log_param("lung_mask_fraction", round(float(mask.float().mean()), 4))
    mlflow.log_params(
        {
            "n_train_normal": len(splits.x_train_normal),
            "n_val": len(splits.y_val),
            "n_val_positive": int(splits.y_val.sum()),
            "n_test": len(y_test),
            "n_test_positive": int(y_test.sum()),
            "n_test_other": len(splits.x_test_other),
            **{
                f"n_excluded_{split}_{reason.replace(' ', '_')}": int(n)
                for split, row in splits.excluded.iterrows()
                for reason, n in row.items()
            },
        }
    )
    paths.reports.mkdir(parents=True, exist_ok=True)
    splits.excluded.to_csv(paths.reports / "excluded.csv", index_label="split")

    # =============================================
    # TREINO DO VAE (somente exames normais)
    # =============================================

    describe_architecture(VAE(latent_dim, image_size=image_size))

    logger.info(f"Treinando o VAE (semente {seeds[0]}, {epochs} épocas)...")
    vae, history = train_autoencoder(
        splits.x_train_normal,
        splits.x_val_normal,
        seed=seeds[0],
        latent_dim=latent_dim,
        epochs=epochs,
    )
    plot_training_curves(history, KL_WARMUP_EPOCHS, paths.figures / "training_curves.png")
    log_history(history)
    pd.DataFrame(history, index=pd.RangeIndex(1, epochs + 1, name="epoch")).to_csv(
        paths.reports / "history.csv"
    )

    # Amostras de z ~ N(0, I) passadas pelo decoder: o VAE também é um modelo gerativo.
    set_seed(123)
    with torch.no_grad():
        samples = vae.decode(torch.randn(16, latent_dim).to(DEVICE)).cpu()
    plot_generated_samples(samples, paths.figures / "generated_samples.png")

    # =============================================
    # CALIBRAÇÃO NA VALIDAÇÃO
    # Escolhe σ e o limiar sem tocar no teste
    # =============================================

    rec_val = reconstruct(vae, splits.x_val)
    sigma, auc_val_by_sigma = select_sigma(splits.x_val, rec_val, splits.y_val, mask=mask)
    score_val = anomaly_score(splits.x_val, rec_val, sigma, mask)
    threshold = calibrate_threshold(score_val, splits.y_val)
    mlflow.log_params({"sigma": sigma, "threshold": threshold})
    mlflow.log_metric("val_auc", auc_val_by_sigma[sigma])
    mlflow.log_metrics({f"val_auc_sigma_{s:g}": auc for s, auc in auc_val_by_sigma.items()})

    # Métricas de validação para comparar configurações (ablações) sem olhar o teste
    maps_val = anomaly_map(splits.x_val, rec_val, sigma, mask)
    val_metrics = {
        "average_precision": float(average_precision_score(splits.y_val, score_val)),
        "ranking": ranking_at_k(splits.y_val, score_val),
        "equipment": evaluate_equipment(splits.y_val, score_val, splits.equipment_val),
        "localization": evaluate_localization(
            maps_val, splits.boxes_val, split="validação", lung_mask=mask
        ),
    }
    mlflow.log_metrics(flatten(val_metrics, "val"))
    pd.DataFrame(
        {
            "label": splits.y_val,
            "view": splits.views_val,
            "equipment": splits.equipment_val,
            "anomaly_score": score_val,
        }
    ).to_csv(paths.reports / "val_predictions.csv", index_label="val_index")

    # =============================================
    # AVALIAÇÃO NO TESTE
    # =============================================

    predictions, rec_test, maps_test = predict(vae, splits.x_test, sigma, threshold, mask)
    score_test = predictions["anomaly_score"].to_numpy()
    metrics = evaluate_model(y_test, score_test, threshold)

    # A mesma curva de σ no teste (só para a figura) mostra se a escolha generaliza.
    # Com σ = 0, o score é o erro pixel a pixel ingênuo, que funciona mal (Seção 7.2).
    auc_test_by_sigma = {
        s: roc_auc_score(y_test, anomaly_score(splits.x_test, rec_test, s, mask))
        for s in auc_val_by_sigma
    }
    metrics["roc_auc_naive"] = auc_test_by_sigma[0]
    logger.info(f"AUC do erro pixel a pixel ingênuo (σ = 0): {metrics['roc_auc_naive']:.3f}")
    plot_sigma_selection(
        auc_val_by_sigma, auc_test_by_sigma, sigma, paths.figures / "sigma_selection.png"
    )

    thresholds = {
        f"p{p} dos normais (sem rótulos)": float(np.percentile(score_val[splits.y_val == 0], p))
        for p in [95, 90, 80]
    }
    thresholds["Youden (com rótulos)"] = youden_threshold(score_val, splits.y_val)
    threshold_table = compare_thresholds(thresholds, splits.y_val, score_val, y_test, score_test)

    queue = simulate_queue(y_test, score_test)
    metrics["ranking"] = ranking_at_k(y_test, score_test)

    plot_test_evaluation(y_test, score_test, paths.figures / "test_evaluation.png", positive)
    plot_gain_curve(y_test, score_test, paths.figures / "gain_curve.png", f"{positive}s")

    # Escala de cor comum a todos os mapas de calor (inclusive os da inferência)
    vmax_map = float(np.percentile(maps_test.numpy(), 99.5))
    label_names = np.where(y_test == 1, positive, "normal")

    rng = np.random.default_rng(42)
    examples = [
        *rng.choice(np.where(y_test == 0)[0], 4, replace=False),
        *rng.choice(np.where(y_test == 1)[0], 4, replace=False),
    ]
    by_score = np.argsort(-score_test)
    false_pos = [i for i in by_score if y_test[i] == 0 and score_test[i] >= threshold][:4]
    false_neg = [i for i in by_score[::-1] if y_test[i] == 1 and score_test[i] < threshold][:4]

    for indices, title, name in [
        (examples, f"4 exames normais e 4 com {positive}", "heatmaps.png"),
        (false_pos, "Falsos positivos mais fortes", "false_positives.png"),
        (false_neg, "Falsos negativos mais fortes", "false_negatives.png"),
    ]:
        if indices:
            plot_heatmaps(
                splits.x_test,
                rec_test,
                maps_test,
                score_test,
                indices,
                vmax_map,
                title,
                paths.figures / name,
                labels=label_names,
                boxes=splits.boxes_test,
            )

    # =============================================
    # OUTRAS ANOMALIAS, LOCALIZAÇÃO E INCIDÊNCIA
    # =============================================

    pred_other, rec_other, maps_other = predict(vae, splits.x_test_other, sigma, threshold, mask)
    score_other = pred_other["anomaly_score"].to_numpy()
    metrics["other_anomalies"] = evaluate_other_anomalies(
        score_test[y_test == 0], score_other, threshold
    )
    plot_heatmaps(
        splits.x_test_other,
        rec_other,
        maps_other,
        score_other,
        list(rng.choice(len(score_other), 6, replace=False)),
        vmax_map,
        "Anormais sem opacidade (o VAE nunca foi ajustado para eles)",
        paths.figures / "other_anomalies.png",
        labels=np.full(len(score_other), "anormal s/ opacidade"),
    )

    metrics["localization"] = evaluate_localization(maps_test, splits.boxes_test, lung_mask=mask)
    metrics["views"] = evaluate_views(y_test, score_test, splits.views_test)
    metrics["equipment"] = evaluate_equipment(y_test, score_test, splits.equipment_test)
    predictions.insert(0, "equipment", splits.equipment_test)
    predictions.insert(0, "view", splits.views_test)

    # =============================================
    # SALVA MODELO E RELATÓRIOS
    # =============================================

    save_model(
        vae,
        metadata={
            "version": version,
            "hygiene": hygiene,
            "views": view_list,
            "image_size": image_size,
            "latent_dim": latent_dim,
            "epochs": epochs,
            "seed": seeds[0],
            "positive_class": positive,
            "sigma": sigma,
            "threshold": threshold,
            "threshold_rule": f"percentil {THRESHOLD_PERCENTILE} dos normais da validação",
            "vmax_map": vmax_map,
            "auc_val": auc_val_by_sigma[sigma],
            **preprocessing.as_dict(),
        },
        model_path=paths.model,
        metadata_path=paths.metadata,
    )
    if mask is not None:
        np.save(paths.model.parent / "lung_mask.npy", mask.numpy())

    predictions.insert(0, "label", y_test)
    predictions.to_csv(paths.reports / "test_predictions.csv", index_label="test_index")
    threshold_table.to_csv(paths.reports / "thresholds.csv")

    # =============================================
    # COMPARAÇÃO COM ALTERNATIVAS
    # =============================================

    if baselines:
        comparison = compare_methods(splits, {seeds[0]: vae}, seeds, epochs=epochs, mask=mask)
        comparison.to_csv(paths.reports / "method_comparison.csv")
        plot_method_comparison(comparison, paths.figures / "method_comparison.png")
        metrics["comparison"] = (
            comparison[["auc_test", "auc_test_std"]].round(4).to_dict(orient="index")
        )
        log_comparison(comparison)

    with open(paths.reports / "metrics.json", "w", encoding="utf-8") as file:
        json.dump({**metrics, "queue_simulation": queue}, file, indent=4, ensure_ascii=False)

    # =============================================
    # MLFLOW: métricas de teste e artefatos
    # =============================================

    test_metrics = {k: v for k, v in metrics.items() if k not in ("threshold", "comparison")}
    mlflow.log_metrics(flatten({**test_metrics, "queue_simulation": queue}, "test"))
    mlflow.log_artifacts(str(paths.reports), artifact_path="reports")
    mlflow.log_artifact(str(paths.model), artifact_path="model")
    mlflow.log_artifact(str(paths.metadata), artifact_path="model")

    logger.success(f"Pipeline executado com sucesso. Relatórios em {paths.reports}")


if __name__ == "__main__":
    app()
