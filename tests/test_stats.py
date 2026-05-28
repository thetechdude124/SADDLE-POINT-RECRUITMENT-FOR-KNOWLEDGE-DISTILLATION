"""Tests for sprkd.stats - McNemar test."""

from math import isclose

import pytest
import torch

from sprkd.stats import (
    McNemarResult,
    mcnemar_table,
    mcnemar_test,
    pairwise_mcnemar,
)


def _toy_predictions():
    """Construct deterministic predictions with a known contingency table.

    Targets: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] (5 of each class)
    A:       [0, 1, 0, 1, 0, 1, 1, 0, 1, 1]   (correct on 6/10)
    B:       [0, 1, 0, 1, 1, 0, 0, 1, 0, 0]   (correct on 6/10)

    Per-element correctness:
       A:  ✓ ✓ ✓ ✓ ✓ ✓ ✗ ✗ ✗ ✓
       B:  ✓ ✓ ✓ ✓ ✗ ✗ ✓ ✓ ✓ ✗

    Contingency cells:
       a (both ✓): indices {0,1,2,3} -> 4
       b (A✓ B✗):  indices {4,5,9} -> 3
       c (A✗ B✓):  indices {6,7,8} -> 3
       d (both ✗): {} -> 0
    """

    targets = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    preds_a = torch.tensor([0, 1, 0, 1, 0, 1, 1, 0, 1, 1])
    preds_b = torch.tensor([0, 1, 0, 1, 1, 0, 0, 1, 0, 0])
    return preds_a, preds_b, targets


def test_mcnemar_table_matches_hand_count():
    a_preds, b_preds, y = _toy_predictions()
    a, b, c, d = mcnemar_table(a_preds, b_preds, y)
    assert (a, b, c, d) == (4, 3, 3, 0)


def test_mcnemar_table_validates_lengths():
    with pytest.raises(ValueError):
        mcnemar_table(torch.zeros(3), torch.zeros(4), torch.zeros(3))


def test_mcnemar_exact_returns_p_value_one_for_balanced_discords():
    a, b, y = _toy_predictions()
    res = mcnemar_test(a, b, y, method="exact")
    assert isinstance(res, McNemarResult)
    assert res.method.startswith("exact")
    assert res.statistic == 3
    assert isclose(res.p_value, 1.0, abs_tol=1e-9)
    assert res.accuracy_a == 0.7
    assert res.accuracy_b == 0.7


def test_mcnemar_exact_zero_discords():
    a = torch.tensor([0, 1, 0, 1])
    b = torch.tensor([0, 1, 0, 1])
    y = torch.tensor([0, 1, 0, 1])
    res = mcnemar_test(a, b, y, method="exact")
    assert res.b == 0 and res.c == 0
    assert res.p_value == 1.0


def test_mcnemar_exact_extreme_difference():
    """All discords favor B -> tiny p-value."""

    n = 50
    targets = torch.zeros(n, dtype=torch.long)
    a = torch.ones(n, dtype=torch.long)        # always wrong
    b = torch.zeros(n, dtype=torch.long)       # always right
    res = mcnemar_test(a, b, targets, method="exact")
    assert res.b == 0 and res.c == n
    assert res.p_value < 1e-12


def test_mcnemar_chi2_continuity_correction():
    a, b, y = _toy_predictions()
    res = mcnemar_test(a, b, y, method="chi2", continuity=True)
    # b == c -> stat is zero -> p == 1
    assert res.statistic == 0.0
    assert res.p_value == 1.0


def test_mcnemar_chi2_matches_textbook_value():
    """Reference: McNemar 1947 example."""

    # Construct preds with contingency (a=10, b=15, c=5, d=20).
    targets = torch.tensor([0] * 50)
    a_preds = torch.tensor([0] * 25 + [1] * 25)        # 25 correct
    b_preds = torch.tensor([0] * 10 + [1] * 15 + [0] * 5 + [1] * 20)
    a, b, c, d = mcnemar_table(a_preds, b_preds, targets)
    assert (a, b, c, d) == (10, 15, 5, 20)
    res = mcnemar_test(a_preds, b_preds, targets, method="chi2", continuity=True)
    # Yates' chi^2 = (|b-c| - 1)^2 / (b+c) = (10-1)^2 / 20 = 4.05
    assert abs(res.statistic - 4.05) < 1e-9
    # p ~ 0.044
    assert 0.03 < res.p_value < 0.05


def test_mcnemar_invalid_method_raises():
    a, b, y = _toy_predictions()
    with pytest.raises(ValueError):
        mcnemar_test(a, b, y, method="bayes")


def test_pairwise_mcnemar_runs_over_three_models():
    a, b, y = _toy_predictions()
    rows = pairwise_mcnemar(
        {"A": a, "B": b, "Identity": y},
        targets=y,
        method="exact",
    )
    assert len(rows) == 3
    pairs = {(r["model_a"], r["model_b"]) for r in rows}
    assert pairs == {("A", "B"), ("A", "Identity"), ("B", "Identity")}
    # Identity vs targets is perfect, so the only b/c contributions are from
    # the non-identity classifier.
    for r in rows:
        if "Identity" in (r["model_a"], r["model_b"]):
            other_acc_key = (
                "accuracy_b" if r["model_a"] == "Identity" else "accuracy_a"
            )
            assert (
                r["accuracy_a" if r["model_a"] == "Identity" else "accuracy_b"]
                == 1.0
            )
            assert 0.5 <= r[other_acc_key] <= 0.9


def test_mcnemar_result_to_dict_round_trip():
    a, b, y = _toy_predictions()
    res = mcnemar_test(a, b, y)
    d = res.to_dict()
    for key in (
        "n",
        "a_correct_b_correct",
        "a_correct_b_wrong",
        "a_wrong_b_correct",
        "a_wrong_b_wrong",
        "statistic",
        "p_value",
        "method",
        "accuracy_a",
        "accuracy_b",
    ):
        assert key in d


def test_mcnemar_use_statsmodels_falls_back_silently_when_missing(monkeypatch):
    """If statsmodels is not importable, the call must still succeed."""

    import builtins

    real_import = builtins.__import__

    def _no_statsmodels(name, *a, **kw):
        if name.startswith("statsmodels"):
            raise ImportError("statsmodels missing in this test")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _no_statsmodels)
    a, b, y = _toy_predictions()
    res = mcnemar_test(a, b, y, method="exact", use_statsmodels=True)
    # falls back to our own implementation
    assert "statsmodels" not in res.method


def test_mcnemar_use_statsmodels_matches_inhouse_p_value():
    """When statsmodels is installed, exact p-values must agree."""

    pytest.importorskip("statsmodels")
    a, b, y = _toy_predictions()

    in_house = mcnemar_test(a, b, y, method="exact", use_statsmodels=False)
    via_sm = mcnemar_test(a, b, y, method="exact", use_statsmodels=True)
    assert via_sm.method.startswith("statsmodels")
    assert via_sm.p_value == pytest.approx(in_house.p_value, abs=1e-9)


def test_mcnemar_use_statsmodels_chi2_matches_inhouse():
    """Chi-squared variant must agree numerically with the canonical Colab path."""

    pytest.importorskip("statsmodels")

    # Same canonical Colab construction: contingency (10, 15, 5, 20).
    targets = torch.tensor([0] * 50)
    a_preds = torch.tensor([0] * 25 + [1] * 25)
    b_preds = torch.tensor([0] * 10 + [1] * 15 + [0] * 5 + [1] * 20)

    in_house = mcnemar_test(a_preds, b_preds, targets, method="chi2", continuity=True)
    via_sm = mcnemar_test(
        a_preds, b_preds, targets, method="chi2", continuity=True, use_statsmodels=True
    )
    assert via_sm.statistic == pytest.approx(in_house.statistic, abs=1e-9)
    assert via_sm.p_value == pytest.approx(in_house.p_value, abs=1e-9)
