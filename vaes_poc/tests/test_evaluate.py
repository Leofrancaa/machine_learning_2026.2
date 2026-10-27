import numpy as np
import pandas as pd
import pytest

from module_vae.modeling.evaluate import evaluate_equipment, metrics_at_threshold, ranking_at_k


def test_f1_and_accuracy_at_threshold():
    y = np.array([0, 0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.9, 0.8, 0.3])

    m = metrics_at_threshold(y, score, threshold=0.5)  # tp = 1, fp = 1, fn = 1, tn = 2

    assert m["f1"] == pytest.approx(0.5)
    assert m["accuracy"] == pytest.approx(3 / 5)


def test_all_normal_reference_has_accuracy_equal_to_share_of_normals():
    y = np.array([0, 0, 0, 0, 1])

    m = metrics_at_threshold(y, np.random.default_rng(0).random(5), threshold=np.inf)

    assert m["accuracy"] == pytest.approx(0.8)
    assert m["f1"] == 0
    assert m["sensitivity"] == 0


def test_ranking_at_k_with_a_perfect_score():
    y = np.r_[np.ones(10), np.zeros(90)].astype(int)
    score = y + np.linspace(0, 0.1, 100)

    r = ranking_at_k(y, score, ks=(10, 25))

    assert r["prevalence"] == pytest.approx(0.1)
    assert r["precision_at_10"] == 1.0 and r["recall_at_10"] == 1.0
    assert r["precision_at_25"] == pytest.approx(10 / 25) and r["recall_at_25"] == 1.0


def test_equipment_auc_only_for_groups_with_enough_positives():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(20), np.zeros(80), np.ones(2), np.zeros(50)].astype(int)
    equipment = np.r_[np.full(100, "0.143"), np.full(52, "0.168")]

    results = evaluate_equipment(y, rng.random(len(y)), equipment, min_positives=10)

    assert results["0.143"]["n"] == 100 and "roc_auc" in results["0.143"]
    assert results["0.168"]["n_positive"] == 2 and "roc_auc" not in results["0.168"]


def test_ablation_rule():
    from module_vae.modeling.compare_runs import AUC, CHANCE, CHANCE_IN_MASK, POINTING, apply_rule

    reference = pd.DataFrame(
        {AUC: [0.75, 0.77, 0.785], POINTING: [0.16, 0.28, 0.25], CHANCE: [0.075] * 3}
    )

    def candidate(auc_gain, pointing_gain, chance_in_mask=None):
        runs = reference + [auc_gain, pointing_gain, 0.0]
        if chance_in_mask is not None:
            runs[CHANCE_IN_MASK] = chance_in_mask
        return runs

    assert apply_rule(reference, candidate(0.025, 0.0))["accepted"]
    assert apply_rule(reference, candidate(0.0, 0.09))["accepted"]
    assert not apply_rule(reference, candidate(0.015, 0.05))["accepted"]  # ganho pequeno
    assert not apply_rule(reference, candidate(0.03, -0.10))["accepted"]  # piora a outra
    # Com a máscara, o acaso sobe 10 pontos: um ganho bruto de 10 pontos não conta.
    assert not apply_rule(reference, candidate(0.0, 0.10, chance_in_mask=0.175))["accepted"]
    assert apply_rule(reference, candidate(0.0, 0.20, chance_in_mask=0.175))["accepted"]
