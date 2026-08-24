"""Loading, integration and persistence of the Olist analytical dataset."""

from pathlib import Path

from loguru import logger
import pandas as pd

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


def load_data(
    orders_path: Path,
    items_path: Path,
    customers_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the raw tables required by the delivery-delay pipeline."""
    paths = (orders_path, items_path, customers_path)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        missing_list = "\n- ".join(missing)
        raise FileNotFoundError(f"Missing Olist raw-data files:\n- {missing_list}")

    orders = pd.read_csv(orders_path, parse_dates=list(ORDER_DATE_COLUMNS))
    items = pd.read_csv(items_path)
    customers = pd.read_csv(customers_path)
    return orders, items, customers


def create_dataset(
    orders: pd.DataFrame,
    items: pd.DataFrame,
    customers: pd.DataFrame,
) -> pd.DataFrame:
    """Integrate delivered orders, aggregated items and customer location."""
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

    items_agg = items.groupby("order_id", as_index=False).agg(
        item_count=("order_item_id", "count"),
        seller_count=("seller_id", "nunique"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
    )

    dataset = delivered.merge(
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

    if not dataset["order_id"].is_unique:
        raise ValueError("The analytical table must contain one row per order.")
    if len(dataset) != len(delivered):
        raise ValueError("The joins changed the number of eligible delivered orders.")

    return dataset


def save_dataset(dataset: pd.DataFrame, output_path: Path) -> None:
    """Persist the intermediate analytical dataset as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    logger.success(f"Intermediate dataset saved to {output_path}")
