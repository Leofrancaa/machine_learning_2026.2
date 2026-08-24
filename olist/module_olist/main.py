"""Executable pipeline that creates the intermediate Olist dataset."""

from loguru import logger

from module_olist.config import INTERIM_DATA_DIR, RAW_DATA_DIR
from module_olist.dataset import create_dataset, load_data, save_dataset
from module_olist.features import create_features

ORDERS_PATH = RAW_DATA_DIR / "olist_orders_dataset.csv"
ITEMS_PATH = RAW_DATA_DIR / "olist_order_items_dataset.csv"
CUSTOMERS_PATH = RAW_DATA_DIR / "olist_customers_dataset.csv"
OUTPUT_PATH = INTERIM_DATA_DIR / "dataset.csv"


def main() -> None:
    """Load raw data, build features and save the intermediate dataset."""
    logger.info("Loading raw Olist tables...")
    orders, items, customers = load_data(ORDERS_PATH, ITEMS_PATH, CUSTOMERS_PATH)

    logger.info("Building the order-level analytical dataset...")
    dataset = create_dataset(orders, items, customers)
    dataset = create_features(dataset)

    save_dataset(dataset, OUTPUT_PATH)
    logger.success(f"Pipeline completed with {len(dataset):,} orders.")


if __name__ == "__main__":
    main()
