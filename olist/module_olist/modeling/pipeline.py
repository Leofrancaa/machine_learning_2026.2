"""Modeling pipelines for delivery-delay classification."""

from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

NUMERIC_FEATURES = [
    "promised_days",
    "item_count",
    "seller_count",
    "total_price",
    "total_freight",
]

CATEGORICAL_FEATURES = [
    "purchase_month",
    "purchase_weekday",
    "purchase_hour",
    "customer_state",
]


def create_preprocessor() -> ColumnTransformer:
    """Create the preprocessing shared by both boosting models."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def create_xgboost_pipeline() -> Pipeline:
    """Create an XGBoost classification pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            (
                "model",
                XGBClassifier(
                    eval_metric="logloss",
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )


def create_lightgbm_pipeline() -> Pipeline:
    """Create a LightGBM classification pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            (
                "model",
                LGBMClassifier(
                    n_jobs=-1,
                    random_state=42,
                    verbosity=-1,
                ),
            ),
        ]
    )


def create_gradient_boosting_pipeline() -> Pipeline:
    """Create a Gradient Boosting classification pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            (
                "model",
                GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=42,
                ),
            ),
        ]
    )
