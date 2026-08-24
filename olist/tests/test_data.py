import pandas as pd
import pytest

import module_olist.main as pipeline
from module_olist.eda import (
    build_order_analysis,
    deadline_risk,
    monthly_risk,
    seller_risk,
    state_risk,
    target_summary,
)


@pytest.fixture
def olist_tables():
    orders = pd.DataFrame(
        {
            "order_id": ["o1", "o2", "o3"],
            "customer_id": ["c1", "c2", "c3"],
            "order_status": ["delivered", "delivered", "canceled"],
            "order_purchase_timestamp": pd.to_datetime(
                ["2018-01-01 10:00", "2018-02-01 20:00", "2018-03-01 08:00"]
            ),
            "order_approved_at": pd.to_datetime(
                ["2018-01-01", "2018-02-01", "2018-03-01"]
            ),
            "order_delivered_customer_date": pd.to_datetime(
                ["2018-01-09", "2018-02-07", None]
            ),
            "order_delivered_carrier_date": pd.to_datetime(
                ["2018-01-03", "2018-02-03", None]
            ),
            "order_estimated_delivery_date": pd.to_datetime(
                ["2018-01-10", "2018-02-05", "2018-03-08"]
            ),
        }
    )
    items = pd.DataFrame(
        {
            "order_id": ["o1", "o1", "o2", "o3"],
            "order_item_id": [1, 2, 1, 1],
            "seller_id": ["s1", "s2", "s1", "s1"],
            "price": [10.0, 20.0, 15.0, 5.0],
            "freight_value": [2.0, 3.0, 4.0, 1.0],
        }
    )
    customers = pd.DataFrame(
        {
            "customer_id": ["c1", "c2", "c3"],
            "customer_state": ["BA", "SP", "RJ"],
            "customer_city": ["salvador", "sao paulo", "rio de janeiro"],
        }
    )
    return orders, items, customers


@pytest.fixture
def analysis(olist_tables):
    return build_order_analysis(*olist_tables)


def test_build_order_analysis_keeps_one_row_per_delivered_order(analysis):
    assert analysis["order_id"].tolist() == ["o1", "o2"]
    assert analysis["order_id"].is_unique
    assert analysis["is_late"].tolist() == [0, 1]
    assert analysis.loc[analysis["order_id"].eq("o1"), "item_count"].item() == 2
    assert analysis.loc[analysis["order_id"].eq("o1"), "seller_count"].item() == 2
    assert analysis.loc[analysis["order_id"].eq("o1"), "total_price"].item() == 30.0
    assert analysis["purchase_month"].tolist() == [1, 2]


def test_eda_summaries_are_consistent(analysis):
    target = target_summary(analysis)
    assert target.loc["Atrasado", "pedidos"] == 1
    assert target.loc["Atrasado", "percentual"] == 50.0

    deadlines = deadline_risk(analysis)
    assert deadlines["pedidos"].sum() == 2
    assert deadlines["taxa_atraso"].between(0, 100).all()

    sellers = seller_risk(analysis)
    assert sellers["pedidos"].sum() == 2

    states = state_risk(analysis, min_orders=1)
    assert set(states["customer_state"]) == {"BA", "SP"}

    months = monthly_risk(analysis)
    assert months["pedidos"].sum() == 2


def test_build_order_analysis_reports_missing_columns(olist_tables):
    orders, items, customers = olist_tables
    with pytest.raises(ValueError, match="order_status"):
        build_order_analysis(orders.drop(columns="order_status"), items, customers)


def test_main_creates_intermediate_dataset(tmp_path, olist_tables, monkeypatch):
    orders, items, customers = olist_tables
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    orders_path = raw_dir / "olist_orders_dataset.csv"
    items_path = raw_dir / "olist_order_items_dataset.csv"
    customers_path = raw_dir / "olist_customers_dataset.csv"
    output_path = tmp_path / "interim" / "dataset.csv"

    orders.to_csv(orders_path, index=False)
    items.to_csv(items_path, index=False)
    customers.to_csv(customers_path, index=False)

    monkeypatch.setattr(pipeline, "ORDERS_PATH", orders_path)
    monkeypatch.setattr(pipeline, "ITEMS_PATH", items_path)
    monkeypatch.setattr(pipeline, "CUSTOMERS_PATH", customers_path)
    monkeypatch.setattr(pipeline, "OUTPUT_PATH", output_path)

    pipeline.main()

    saved = pd.read_csv(output_path)
    assert saved["order_id"].tolist() == ["o1", "o2"]
    assert {"is_late", "promised_days", "purchase_month"}.issubset(saved.columns)
