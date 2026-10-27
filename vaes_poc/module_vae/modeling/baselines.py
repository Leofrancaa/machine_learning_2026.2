"""Comparação do VAE com alternativas.

Sem rótulos da doença (mesmas condições do VAE): brilho médio, imagem média dos
normais, PCA com 8 componentes e autoencoder comum. Com rótulos (referências, condições
mais favoráveis): heurística de nível de detalhe e CNN supervisionada.

Todos os métodos baseados em reconstrução usam o mesmo score suavizado, com σ escolhido
na validação para cada método.
"""

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
import torch

from module_vae.config import DEVICE, EPOCHS
from module_vae.features import anomaly_score, brightness, detail_level
from module_vae.modeling.architecture import VAE
from module_vae.modeling.calibration import sigmas_for
from module_vae.modeling.predict import reconstruct
from module_vae.modeling.split import DataSplits
from module_vae.modeling.train import train_autoencoder, train_supervised_cnn


def evaluate_reconstructions(
    splits: DataSplits,
    rec_val: torch.Tensor,
    rec_test: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> dict:
    """Escolhe σ pela AUC de validação e avalia o score no teste com esse σ."""
    auc_val = {
        s: roc_auc_score(splits.y_val, anomaly_score(splits.x_val, rec_val, s, mask))
        for s in sigmas_for(splits.x_val.shape[-1])
    }
    sigma = max(auc_val, key=auc_val.get)
    score_test = anomaly_score(splits.x_test, rec_test, sigma, mask)

    return {
        "sigma": sigma,
        "auc_val": auc_val[sigma],
        "auc_test": roc_auc_score(splits.y_test, score_test),
        "ap_test": average_precision_score(splits.y_test, score_test),
    }


def evaluate_scores(splits: DataSplits, score_val: np.ndarray, score_test: np.ndarray) -> dict:
    """Avalia um score que não depende de reconstrução (não há σ para escolher)."""
    return {
        "sigma": None,
        "auc_val": roc_auc_score(splits.y_val, score_val),
        "auc_test": roc_auc_score(splits.y_test, score_test),
        "ap_test": average_precision_score(splits.y_test, score_test),
    }


@torch.no_grad()
def predict_logits(model: torch.nn.Module, x: torch.Tensor, batch_size: int = 256) -> np.ndarray:
    model.eval()
    return torch.cat(
        [
            model(x[i : i + batch_size].to(DEVICE)).squeeze(1).cpu()
            for i in range(0, len(x), batch_size)
        ]
    ).numpy()


def summarize(results: dict[str, list[dict]]) -> pd.DataFrame:
    """Resume cada método em média ± desvio-padrão das sementes."""
    rows = []
    for name, runs in results.items():
        df = pd.DataFrame(runs)
        rows.append(
            {
                "method": name,
                "uses_labels": name.endswith("*"),
                "sigma": ", ".join(str(s) for s in df["sigma"].dropna().unique()) or None,
                "auc_val": df["auc_val"].mean(),
                "auc_test": df["auc_test"].mean(),
                "auc_test_std": df["auc_test"].std() if len(df) > 1 else 0.0,
                "ap_test": df["ap_test"].mean(),
                "n_runs": len(df),
            }
        )

    return pd.DataFrame(rows).set_index("method")


def compare_methods(
    splits: DataSplits,
    vae_models: dict[int, VAE],
    seeds: list[int],
    epochs: int = EPOCHS,
    mask: torch.Tensor | None = None,
) -> pd.DataFrame:
    """Treina as alternativas e compara todas com o VAE no mesmo protocolo.

    Args:
        splits (DataSplits): Conjuntos de dados.
        vae_models (dict[int, VAE]): VAEs já treinados, por semente. As sementes que
            faltarem em seeds são treinadas aqui.
        seeds (list[int]): Sementes do VAE e do autoencoder comum.
        epochs (int): Épocas de treino do VAE e do autoencoder comum.
        mask (torch.Tensor | None): Máscara dos pulmões (ablação C1), usada no score de
            todos os métodos baseados em reconstrução.

    Returns:
        pd.DataFrame: Uma linha por método. Os marcados com * usam rótulos da doença.
    """
    results = {}
    scale = splits.x_val.shape[-1] / 64  # o σ da heurística de detalhe é definido em 64 px
    # PCA, autoencoder comum e os demais VAEs usam o mesmo tamanho de código do VAE principal
    latent_dim = next(iter(vae_models.values())).latent_dim

    logger.info("Baseline: brilho médio")
    results["Brilho médio"] = [
        evaluate_scores(splits, brightness(splits.x_val), brightness(splits.x_test))
    ]

    logger.info("Baseline: imagem média dos normais como 'reconstrução'")
    mean_image = splits.x_train_normal.mean(0, keepdim=True)
    results["Imagem média"] = [
        evaluate_reconstructions(
            splits,
            mean_image.expand_as(splits.x_val),
            mean_image.expand_as(splits.x_test),
            mask,
        )
    ]

    logger.info(f"Baseline: PCA com {latent_dim} componentes")
    pca = PCA(latent_dim).fit(splits.x_train_normal.flatten(1).numpy())

    def reconstruct_pca(x: torch.Tensor) -> torch.Tensor:
        x_rec = pca.inverse_transform(pca.transform(x.flatten(1).numpy()))
        return torch.tensor(x_rec, dtype=torch.float32).view_as(x)

    results[f"PCA ({latent_dim} componentes)"] = [
        evaluate_reconstructions(
            splits, reconstruct_pca(splits.x_val), reconstruct_pca(splits.x_test), mask
        )
    ]

    ae_models = {}
    for seed in seeds:
        logger.info(f"Treinando autoencoder comum (semente {seed})...")
        ae_models[seed], _ = train_autoencoder(
            splits.x_train_normal,
            splits.x_val_normal,
            seed=seed,
            latent_dim=latent_dim,
            deterministic=True,
            epochs=epochs,
            verbose=False,
        )
        if seed not in vae_models:
            logger.info(f"Treinando VAE (semente {seed})...")
            vae_models[seed], _ = train_autoencoder(
                splits.x_train_normal,
                splits.x_val_normal,
                seed=seed,
                latent_dim=latent_dim,
                epochs=epochs,
                verbose=False,
            )

    for name, models in [("Autoencoder comum", ae_models), ("VAE", vae_models)]:
        results[name] = [
            evaluate_reconstructions(
                splits, reconstruct(m, splits.x_val), reconstruct(m, splits.x_test), mask
            )
            for seed, m in models.items()
            if seed in seeds
        ]

    # Referências que usam rótulos da doença (marcadas com *)
    results["Nível de detalhe*"] = [
        evaluate_scores(
            splits,
            -detail_level(splits.x_val, sigma=1.5 * scale),
            -detail_level(splits.x_test, sigma=1.5 * scale),
        )
    ]

    logger.info("Treinando a CNN supervisionada (usa os rótulos da doença)...")
    cnn = train_supervised_cnn(splits.x_train, splits.y_train)
    results["CNN supervisionada*"] = [
        evaluate_scores(
            splits, predict_logits(cnn, splits.x_val), predict_logits(cnn, splits.x_test)
        )
    ]

    table = summarize(results)
    logger.success(
        f"COMPARAÇÃO DE MÉTODOS (* usa rótulos da doença)\n{table.round(3).to_string()}"
    )

    return table
