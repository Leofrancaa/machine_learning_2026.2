import pandas as pd

from module_olist.modeling.active_learning import run_active_learning


def test_active_learning_queries_only_training_pool_examples():
    rows = 40
    features = pd.DataFrame(
        {
            "promised_days": [10.0 + index % 5 for index in range(rows)],
            "item_count": [1] * rows,
            "seller_count": [1] * rows,
            "total_price": [100.0 + index for index in range(rows)],
            "total_freight": [None if index == 3 else 12.0 for index in range(rows)],
            "purchase_month": [1 + index % 12 for index in range(rows)],
            "purchase_weekday": [index % 7 for index in range(rows)],
            "purchase_hour": [index % 24 for index in range(rows)],
            "customer_state": ["SP" if index % 2 else "RJ" for index in range(rows)],
        },
        index=range(100, 100 + rows),
    )
    target = pd.Series([index % 2 for index in range(rows)], index=features.index)

    result = run_active_learning(
        features, target, initial_labeled=6, query_size=3, max_iterations=2,
        pool_size=30, random_state=42,
    )

    assert result.history["labeled_count"].tolist() == [6, 9, 12]
    assert result.history["unlabeled_count"].tolist() == [24, 21, 18]
    assert len(result.selections) == 6
    assert set(result.labeled_indices).issubset(features.index)
