"""Order-level exploratory analysis for the Olist delivery-delay problem."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from module_olist.config import RAW_DATA_DIR

FILE_NAMES: Mapping[str, str] = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "customers": "olist_customers_dataset.csv",
}

ORDER_DATE_COLUMNS = (
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
)


def _require_columns(frame: pd.DataFrame, columns: set[str], frame_name: str) -> None:
    """Raise a useful error when an input table does not match the expected schema."""
    missing = columns.difference(frame.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"{frame_name} is missing required columns: {missing_list}")


def load_olist_tables(
    raw_data_dir: Path = RAW_DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three raw Olist tables used by the order-delay EDA."""
    paths = {name: raw_data_dir / filename for name, filename in FILE_NAMES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        missing_list = "\n- ".join(missing)
        raise FileNotFoundError(f"Missing Olist raw-data files:\n- {missing_list}")

    orders = pd.read_csv(paths["orders"], parse_dates=list(ORDER_DATE_COLUMNS))
    items = pd.read_csv(paths["items"])
    customers = pd.read_csv(paths["customers"])
    return orders, items, customers


def build_order_analysis(
    orders: pd.DataFrame,
    items: pd.DataFrame,
    customers: pd.DataFrame,
) -> pd.DataFrame:
    """Build a leakage-aware analytical table with exactly one row per delivered order."""
    _require_columns(
        orders,
        {
            "order_id",
            "customer_id",
            "order_status",
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        },
        "orders",
    )
    _require_columns(
        items,
        {
            "order_id",
            "order_item_id",
            "seller_id",
            "price",
            "freight_value",
        },
        "items",
    )
    _require_columns(
        customers,
        {"customer_id", "customer_state", "customer_city"},
        "customers",
    )

    delivered = orders.loc[
        orders["order_status"].eq("delivered")
        & orders["order_delivered_customer_date"].notna()
        & orders["order_estimated_delivery_date"].notna()
        & orders["order_approved_at"].notna()
    ].copy()
    delivered["is_late"] = (
        delivered["order_delivered_customer_date"] > delivered["order_estimated_delivery_date"]
    ).astype("int8")

    items_agg = (
        items.groupby("order_id", as_index=False)
        .agg(
            item_count=("order_item_id", "count"),
            seller_count=("seller_id", "nunique"),
            total_price=("price", "sum"),
            total_freight=("freight_value", "sum"),
        )
        .reset_index(drop=True)
    )

    analysis = delivered.merge(
        items_agg,
        on="order_id",
        how="left",
        validate="one_to_one",
    ).merge(
        customers[["customer_id", "customer_state", "customer_city"]],
        on="customer_id",
        how="left",
        validate="many_to_one",
    )

    if not analysis["order_id"].is_unique:
        raise ValueError("The analytical table must contain one row per order.")
    if len(analysis) != len(delivered):
        raise ValueError("The joins changed the number of eligible delivered orders.")

    analysis["promised_days"] = (
        (analysis["order_estimated_delivery_date"] - analysis["order_approved_at"])
        .dt.total_seconds()
        .div(86_400)
    )
    analysis["purchase_month"] = analysis["order_purchase_timestamp"].dt.month
    analysis["purchase_weekday"] = analysis["order_purchase_timestamp"].dt.dayofweek
    analysis["purchase_hour"] = analysis["order_purchase_timestamp"].dt.hour
    return analysis


def target_summary(analysis: pd.DataFrame) -> pd.DataFrame:
    """Return counts and percentages for on-time and late deliveries."""
    summary = (
        analysis["is_late"]
        .map({0: "No prazo", 1: "Atrasado"})
        .value_counts()
        .reindex(["No prazo", "Atrasado"], fill_value=0)
        .rename_axis("situacao")
        .to_frame("pedidos")
    )
    summary["percentual"] = (summary["pedidos"] / summary["pedidos"].sum() * 100).round(2)
    return summary


def deadline_risk(analysis: pd.DataFrame) -> pd.DataFrame:
    """Compare late-delivery rates across promised-deadline bands."""
    valid = analysis.loc[analysis["promised_days"].gt(0)].copy()
    valid["promised_days_group"] = pd.cut(
        valid["promised_days"],
        bins=[0, 7, 14, 21, 30, np.inf],
        labels=["Até 7", "8–14", "15–21", "22–30", "Mais de 30"],
    )
    result = (
        valid.groupby("promised_days_group", observed=True)
        .agg(pedidos=("order_id", "size"), taxa_atraso=("is_late", "mean"))
        .reset_index()
    )
    result["taxa_atraso"] *= 100
    return result


def complexity_summary(analysis: pd.DataFrame) -> pd.DataFrame:
    """Compare median order-complexity measures by target class."""
    return (
        analysis.groupby("is_late")[
            [
                "item_count",
                "seller_count",
                "total_price",
                "total_freight",
            ]
        ]
        .median()
        .rename(index={0: "No prazo", 1: "Atrasado"})
    )


def seller_risk(analysis: pd.DataFrame) -> pd.DataFrame:
    """Compare risk between single-seller and multi-seller orders."""
    result = (
        analysis.assign(
            seller_group=np.where(analysis["seller_count"].gt(1), "Mais de um", "Um vendedor")
        )
        .groupby("seller_group")
        .agg(pedidos=("order_id", "size"), taxa_atraso=("is_late", "mean"))
        .reset_index()
    )
    result["taxa_atraso"] *= 100
    return result


def state_risk(analysis: pd.DataFrame, min_orders: int = 100) -> pd.DataFrame:
    """Rank customer states by late-delivery rate using a minimum sample size."""
    result = (
        analysis.groupby("customer_state")
        .agg(pedidos=("order_id", "size"), taxa_atraso=("is_late", "mean"))
        .query("pedidos >= @min_orders")
        .sort_values("taxa_atraso", ascending=False)
        .reset_index()
    )
    result["taxa_atraso"] *= 100
    return result


def monthly_risk(analysis: pd.DataFrame) -> pd.DataFrame:
    """Summarize order volume and late-delivery rate by purchase month."""
    result = (
        analysis.groupby("purchase_month")
        .agg(pedidos=("order_id", "size"), taxa_atraso=("is_late", "mean"))
        .reset_index()
    )
    result["taxa_atraso"] *= 100
    return result
