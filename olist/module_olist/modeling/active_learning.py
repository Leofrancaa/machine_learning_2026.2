"""Simulate uncertainty sampling for the Olist delivery-delay problem."""

from dataclasses import dataclass

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import entropy
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.semi_supervised import LabelSpreading

from module_olist.config import INTERIM_DATA_DIR
from module_olist.modeling.pipeline import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from module_olist.modeling.split import split_data


@dataclass
class ActiveLearningResult:
    history: pd.DataFrame
    selections: pd.DataFrame
    labeled_indices: np.ndarray


def run_active_learning(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    initial_labeled: int = 10,
    query_size: int = 5,
    max_iterations: int = 4,
    pool_size: int = 1_000,
    random_state: int = 42,
) -> ActiveLearningResult:
    """Query the most uncertain examples using labels from a simulated oracle."""
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows.")
    if initial_labeled < 2 or query_size < 1 or max_iterations < 0 or pool_size < 2:
        raise ValueError("Active-learning parameters are outside their valid ranges.")
    if pd.Series(y).nunique() != 2:
        raise ValueError("Active learning requires a binary target with both classes.")

    pool_count = min(pool_size, len(X))
    if pool_count < initial_labeled:
        raise ValueError("The data pool is smaller than initial_labeled.")

    positions = np.arange(len(X))
    if pool_count < len(X):
        positions, _ = train_test_split(
            positions, train_size=pool_count, random_state=random_state, stratify=y
        )
    X_pool = X.iloc[positions].reset_index(drop=True)
    y_pool = y.iloc[positions].to_numpy()
    source_indices = np.asarray(X.index)[positions]
    labeled, _ = train_test_split(
        np.arange(pool_count),
        train_size=initial_labeled,
        random_state=random_state,
        stratify=y_pool,
    )
    unlabeled = np.ones(pool_count, dtype=bool)
    unlabeled[labeled] = False

    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("encoder", OneHotEncoder(handle_unknown="ignore"))]),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    features = preprocessor.fit_transform(X_pool)
    if sparse.issparse(features):
        features = features.toarray()

    history = []
    selections = []
    for iteration in range(max_iterations + 1):
        labels = np.full(pool_count, -1, dtype=int)
        labels[labeled] = y_pool[labeled]
        model = LabelSpreading(gamma=0.25, max_iter=20)
        model.fit(features, labels)

        remaining = np.flatnonzero(unlabeled)
        accuracy = (
            accuracy_score(y_pool[remaining], model.transduction_[remaining])
            if len(remaining)
            else np.nan
        )
        history.append(
            {
                "iteration": iteration,
                "labeled_count": len(labeled),
                "unlabeled_count": len(remaining),
                "unlabeled_accuracy": accuracy,
            }
        )
        logger.info(
            "Iteration {}: {} labeled, {} unlabeled, simulated accuracy {:.3f}",
            iteration, len(labeled), len(remaining), accuracy,
        )
        if iteration == max_iterations or not len(remaining):
            break

        uncertainty = entropy(model.label_distributions_[remaining], axis=1)
        chosen_offsets = np.argsort(uncertainty)[-query_size:][::-1]
        chosen = remaining[chosen_offsets]
        for position, offset in zip(chosen, chosen_offsets):
            selections.append(
                {
                    "iteration": iteration + 1,
                    "sample_index": source_indices[position],
                    "uncertainty": uncertainty[offset],
                    "predicted_label": model.transduction_[position],
                    "oracle_label": y_pool[position],
                }
            )
        unlabeled[chosen] = False
        labeled = np.concatenate((labeled, chosen))

    return ActiveLearningResult(
        pd.DataFrame(history), pd.DataFrame(selections), source_indices[labeled]
    )


def plot_active_learning_results(result: ActiveLearningResult) -> None:
    """Plot the examples selected at each query round."""
    if result.selections.empty:
        logger.warning("No examples were selected for plotting.")
        return

    iterations = result.selections["iteration"].unique()
    figure, axes = plt.subplots(
        len(iterations), 1, figsize=(11, max(3, 2.8 * len(iterations))), squeeze=False
    )
    for axis, iteration in zip(axes.flat, iterations):
        selected = result.selections.loc[
            result.selections["iteration"].eq(iteration)
        ].sort_values("uncertainty")
        axis.barh(selected["sample_index"].astype(str), selected["uncertainty"])
        axis.set(title=f"Query {iteration}: most uncertain examples", xlabel="Label entropy")
    figure.tight_layout()
    plt.show()


def main() -> None:
    dataset_path = INTERIM_DATA_DIR / "dataset.csv"
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Prepared dataset not found at {dataset_path}.")
    data = pd.read_csv(dataset_path)
    X_train, _, y_train, _ = split_data(data)
    result = run_active_learning(X_train, y_train)
    logger.info("Active-learning history:\n{}", result.history.to_string(index=False))
    plot_active_learning_results(result)


if __name__ == "__main__":
    main()
