"""Order-level exploratory analysis for the Olist delivery-delay problem."""

from pathlib import Path

import numpy as np
import pandas as pd

from module_olist.config import RAW_DATA_DIR
from module_olist.dataset import create_dataset, load_data
from module_olist.features import create_features


def load_olist_tables(
    raw_data_dir: Path = RAW_DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three raw Olist tables used by the order-delay EDA."""
    return load_data(
        raw_data_dir / "olist_orders_dataset.csv",
        raw_data_dir / "olist_order_items_dataset.csv",
        raw_data_dir / "olist_customers_dataset.csv",
    )


def build_order_analysis(
    orders: pd.DataFrame,
    items: pd.DataFrame,
    customers: pd.DataFrame,
) -> pd.DataFrame:
    """Build a leakage-aware analytical table with exactly one row per delivered order."""
    return create_features(create_dataset(orders, items, customers))


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
