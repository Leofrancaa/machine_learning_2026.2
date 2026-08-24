"""Feature engineering for the Olist delivery-delay problem."""

import pandas as pd


def create_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """Create features available at the payment-approval prediction time."""
    required = {
        "order_purchase_timestamp",
        "order_approved_at",
        "order_estimated_delivery_date",
    }
    missing = required.difference(dataset.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"dataset is missing required columns: {missing_list}")

    features = dataset.copy()
    features["promised_days"] = (
        (features["order_estimated_delivery_date"] - features["order_approved_at"])
        .dt.total_seconds()
        .div(86_400)
    )
    features["purchase_month"] = features["order_purchase_timestamp"].dt.month
    features["purchase_weekday"] = features["order_purchase_timestamp"].dt.dayofweek
    features["purchase_hour"] = features["order_purchase_timestamp"].dt.hour
    return features
