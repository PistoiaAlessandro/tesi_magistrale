"""
Group fairness and profit metrics for credit scoring.

Standalone Python implementation of the metrics used in the thesis
"Fairness and Profitability in Machine Learning Credit Scoring".

Conventions
-----------
y_true : 1 = good payer, 0 = bad payer (default)
y_pred : 1 = loan accepted, 0 = loan rejected
group  : 0 = unprivileged (age < 25), 1 = privileged (age >= 25)

All fairness metrics are absolute differences between groups:
0 means perfectly fair, larger values mean more disparity.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "group_rates",
    "independence",
    "disparate_impact",
    "separation",
    "sufficiency",
    "compute_profit",
    "fairness_report",
]


def _as_arrays(*arrays):
    out = [np.asarray(a).astype(int).ravel() for a in arrays]
    n = len(out[0])
    if any(len(a) != n for a in out):
        raise ValueError("All inputs must have the same length.")
    return out


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den > 0 else float("nan")


def group_rates(y_true, y_pred, group) -> dict[int, dict[str, float]]:
    """Confusion-matrix rates for each group (0 and 1)."""
    y_true, y_pred, group = _as_arrays(y_true, y_pred, group)
    rates: dict[int, dict[str, float]] = {}
    for g in (0, 1):
        m = group == g
        yt, yp = y_true[m], y_pred[m]
        tp = np.sum((yt == 1) & (yp == 1))
        fp = np.sum((yt == 0) & (yp == 1))
        tn = np.sum((yt == 0) & (yp == 0))
        fn = np.sum((yt == 1) & (yp == 0))
        rates[g] = {
            "n": int(m.sum()),
            "acceptance_rate": _safe_div(tp + fp, m.sum()),
            "tpr": _safe_div(tp, tp + fn),
            "fpr": _safe_div(fp, fp + tn),
            "fnr": _safe_div(fn, tp + fn),
            "ppv": _safe_div(tp, tp + fp),
            "npv": _safe_div(tn, tn + fn),
        }
    return rates


def independence(y_true, y_pred, group) -> float:
    """IND = |P(Y^=1 | A=0) - P(Y^=1 | A=1)|  (demographic parity gap)."""
    r = group_rates(y_true, y_pred, group)
    return abs(r[0]["acceptance_rate"] - r[1]["acceptance_rate"])


def disparate_impact(y_true, y_pred, group) -> float:
    """DI = P(Y^=1 | A=0) / P(Y^=1 | A=1). The 80% rule requires DI >= 0.8."""
    r = group_rates(y_true, y_pred, group)
    return _safe_div(r[0]["acceptance_rate"], r[1]["acceptance_rate"])


def separation(y_true, y_pred, group) -> float:
    """SP = 0.5 * (|FPR_0 - FPR_1| + |FNR_0 - FNR_1|)  (equalized odds gap)."""
    r = group_rates(y_true, y_pred, group)
    return 0.5 * (abs(r[0]["fpr"] - r[1]["fpr"]) + abs(r[0]["fnr"] - r[1]["fnr"]))


def sufficiency(y_true, y_pred, group, include_npv: bool = False) -> float:
    """SF = |PPV_0 - PPV_1|  (predictive parity gap).

    With include_npv=True returns 0.5 * (|PPV_0 - PPV_1| + |NPV_0 - NPV_1|).
    """
    r = group_rates(y_true, y_pred, group)
    ppv_gap = abs(r[0]["ppv"] - r[1]["ppv"])
    if not include_npv:
        return ppv_gap
    return 0.5 * (ppv_gap + abs(r[0]["npv"] - r[1]["npv"]))


def compute_profit(y_true, y_pred, amount=None, roi: float = 0.2644, lgd: float = 1.0) -> dict[str, float]:
    """Realised lending profit of a set of decisions.

    Each accepted good loan earns ``roi * amount``; each accepted bad loan
    loses ``lgd * amount``. Rejected applications contribute nothing.

    Returns total profit, average profit per applicant and profit per euro lent.
    """
    y_true, y_pred = _as_arrays(y_true, y_pred)
    amount = np.ones(len(y_true)) if amount is None else np.asarray(amount, dtype=float).ravel()
    if len(amount) != len(y_true):
        raise ValueError("amount must have the same length as y_true.")

    accepted = y_pred == 1
    gains = roi * amount * (accepted & (y_true == 1))
    losses = lgd * amount * (accepted & (y_true == 0))
    total = float(gains.sum() - losses.sum())
    lent = float(amount[accepted].sum())
    return {
        "total_profit": total,
        "avg_profit": total / len(y_true) if len(y_true) else float("nan"),
        "profit_per_euro_lent": _safe_div(total, lent),
    }


def fairness_report(y_true, y_pred, group) -> dict[str, float]:
    """All fairness metrics in a single dictionary."""
    return {
        "IND": independence(y_true, y_pred, group),
        "DI": disparate_impact(y_true, y_pred, group),
        "SP": separation(y_true, y_pred, group),
        "SF": sufficiency(y_true, y_pred, group),
    }


if __name__ == "__main__":
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 1])
    age_group = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    print(fairness_report(y_true, y_pred, age_group))
    print(compute_profit(y_true, y_pred))
