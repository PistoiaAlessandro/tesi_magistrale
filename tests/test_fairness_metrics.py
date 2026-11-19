import math

import numpy as np
import pytest

from src.fairness_metrics import (
    compute_profit,
    disparate_impact,
    fairness_report,
    group_rates,
    independence,
    separation,
    sufficiency,
)

# group 0: y_true 1 0 1 1 | y_pred 1 0 1 0  -> TP=2 FP=0 TN=1 FN=1
# group 1: y_true 0 1 0 1 | y_pred 0 1 1 1  -> TP=2 FP=1 TN=1 FN=0
Y_TRUE = np.array([1, 0, 1, 1, 0, 1, 0, 1])
Y_PRED = np.array([1, 0, 1, 0, 0, 1, 1, 1])
GROUP = np.array([0, 0, 0, 0, 1, 1, 1, 1])


def test_group_rates():
    r = group_rates(Y_TRUE, Y_PRED, GROUP)
    assert r[0]["acceptance_rate"] == pytest.approx(0.5)
    assert r[1]["acceptance_rate"] == pytest.approx(0.75)
    assert r[0]["fpr"] == pytest.approx(0.0)
    assert r[1]["fpr"] == pytest.approx(0.5)
    assert r[0]["fnr"] == pytest.approx(1 / 3)
    assert r[1]["fnr"] == pytest.approx(0.0)
    assert r[0]["ppv"] == pytest.approx(1.0)
    assert r[1]["ppv"] == pytest.approx(2 / 3)


def test_independence_and_di():
    assert independence(Y_TRUE, Y_PRED, GROUP) == pytest.approx(0.25)
    assert disparate_impact(Y_TRUE, Y_PRED, GROUP) == pytest.approx(0.5 / 0.75)


def test_separation():
    assert separation(Y_TRUE, Y_PRED, GROUP) == pytest.approx(0.5 * (0.5 + 1 / 3))


def test_sufficiency():
    assert sufficiency(Y_TRUE, Y_PRED, GROUP) == pytest.approx(1 / 3)
    # NPV: group0 = 1/2, group1 = 1/1
    assert sufficiency(Y_TRUE, Y_PRED, GROUP, include_npv=True) == pytest.approx(0.5 * (1 / 3 + 0.5))


def test_perfectly_fair_classifier():
    y = np.array([1, 0, 1, 0])
    g = np.array([0, 0, 1, 1])
    rep = fairness_report(y, y, g)
    assert rep["IND"] == 0 and rep["SP"] == 0 and rep["SF"] == 0
    assert rep["DI"] == pytest.approx(1.0)


def test_compute_profit_unit_amounts():
    p = compute_profit(Y_TRUE, Y_PRED, roi=0.2644, lgd=1.0)
    # accepted: 5 (4 good, 1 bad)
    assert p["total_profit"] == pytest.approx(4 * 0.2644 - 1)
    assert p["profit_per_euro_lent"] == pytest.approx((4 * 0.2644 - 1) / 5)
    assert p["avg_profit"] == pytest.approx((4 * 0.2644 - 1) / 8)


def test_compute_profit_with_amounts():
    p = compute_profit([1, 0, 1], [1, 1, 0], amount=[1000, 500, 2000], roi=0.2, lgd=0.5)
    assert p["total_profit"] == pytest.approx(200 - 250)
    assert p["profit_per_euro_lent"] == pytest.approx(-50 / 1500)


def test_no_acceptances_gives_nan_profit_per_euro():
    p = compute_profit([1, 0], [0, 0])
    assert p["total_profit"] == 0
    assert math.isnan(p["profit_per_euro_lent"])


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        independence([1, 0], [1], [0, 1])
