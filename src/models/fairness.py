"""
Fairness evaluation module - Task 4.

Implements two standard fairness metrics used in fair lending research:

1. Demographic Parity Difference (DPD)
   Measures whether the model approves loans at the same rate for all groups.
   DPD = max_group(P(Ŷ=1)) - min_group(P(Ŷ=1))
   Ideal value: 0  (same approval rate across groups)

2. Equalized Odds Difference (EOD)
   Measures whether the model's error rates (TPR, FPR) are equal across groups.
   EOD = max(ΔTPR, ΔFPR) where Δ is the range across groups
   Ideal value: 0  (same error rates across groups)

Protected attributes in the German Credit dataset:
  - personal_status_sex  (gender × marital status combinations)
  - age / age_group      (continuous or binned)

Usage
-----
    from src.models.fairness import demographic_parity, equalized_odds, fairness_report
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def demographic_parity(
    y_pred: np.ndarray,
    protected: np.ndarray,
) -> tuple[float, dict]:
    """
    Demographic Parity Difference (DPD).

    Parameters
    ----------
    y_pred     : binary predictions (0/1)
    protected  : group labels (any dtype, e.g. encoded int or string)

    Returns
    -------
    dpd              : float – range of positive-prediction rates across groups
    group_rates      : dict  – {group_label: positive_rate}
    """
    groups = np.unique(protected)
    group_rates: dict = {}
    for g in groups:
        mask = protected == g
        group_rates[str(g)] = round(float(y_pred[mask].mean()), 4)

    rates = list(group_rates.values())
    dpd = round(max(rates) - min(rates), 4) if len(rates) >= 2 else 0.0
    return dpd, group_rates


def equalized_odds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected: np.ndarray,
) -> tuple[float, dict]:
    """
    Equalized Odds Difference (EOD).

    Parameters
    ----------
    y_true    : ground-truth labels (0/1)
    y_pred    : binary predictions  (0/1)
    protected : group labels

    Returns
    -------
    eod           : float – max(ΔTPR, ΔFPR) across groups
    group_metrics : dict  – {group: {tpr, fpr, support}}
    """
    groups = np.unique(protected)
    group_metrics: dict = {}

    for g in groups:
        mask = protected == g
        yt = y_true[mask]
        yp = y_pred[mask]

        tp = int(((yt == 1) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        tn = int(((yt == 0) & (yp == 0)).sum())

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        group_metrics[str(g)] = {
            "tpr": round(tpr, 4),
            "fpr": round(fpr, 4),
            "support": int(mask.sum()),
        }

    tprs = [m["tpr"] for m in group_metrics.values()]
    fprs = [m["fpr"] for m in group_metrics.values()]

    eod = 0.0
    if len(tprs) >= 2:
        eod = round(max(max(tprs) - min(tprs), max(fprs) - min(fprs)), 4)

    return eod, group_metrics


# ---------------------------------------------------------------------------
# Composite report
# ---------------------------------------------------------------------------

def fairness_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X_test: pd.DataFrame,
    protected_col: str,
    model_name: str = "model",
) -> dict:
    """
    Generate a full fairness report for one protected attribute.

    Parameters
    ----------
    y_true        : ground-truth labels
    y_pred        : binary predictions
    X_test        : test feature dataframe (must contain protected_col)
    protected_col : column name of the protected attribute
    model_name    : label used in the report

    Returns
    -------
    report : dict with DPD, EOD, per-group metrics, and a plain-text summary
    """
    if protected_col not in X_test.columns:
        return {"error": f"Column '{protected_col}' not found in X_test."}

    protected = X_test[protected_col].values

    dpd, group_rates = demographic_parity(y_pred, protected)
    eod, group_metrics = equalized_odds(y_true, y_pred, protected)

    # Merge per-group stats
    groups_combined: dict = {}
    for g in group_rates:
        groups_combined[g] = {
            "positive_prediction_rate": group_rates[g],
            **group_metrics.get(g, {}),
        }

    # Bias flag thresholds (common regulatory guidance: DPD ≤ 0.10)
    dpd_flag = dpd > 0.10
    eod_flag = eod > 0.10

    summary_lines = [
        f"Fairness Report — {model_name} | Protected attribute: {protected_col}",
        "-" * 60,
        f"  Demographic Parity Difference : {dpd:.4f}  {'⚠ BIAS DETECTED' if dpd_flag else '✓ OK'}",
        f"  Equalized Odds Difference     : {eod:.4f}  {'⚠ BIAS DETECTED' if eod_flag else '✓ OK'}",
        "",
        "  Per-group breakdown:",
    ]
    for g, m in groups_combined.items():
        summary_lines.append(
            f"    [{g}]  approval_rate={m.get('positive_prediction_rate', 'N/A'):.4f}  "
            f"TPR={m.get('tpr', 'N/A'):.4f}  FPR={m.get('fpr', 'N/A'):.4f}  "
            f"n={m.get('support', 'N/A')}"
        )

    return {
        "model": model_name,
        "protected_attribute": protected_col,
        "demographic_parity_difference": dpd,
        "equalized_odds_difference": eod,
        "bias_flags": {"dpd": dpd_flag, "eod": eod_flag},
        "group_metrics": groups_combined,
        "summary": "\n".join(summary_lines),
    }


# ---------------------------------------------------------------------------
# Age binariser helper (young ≤ 25 vs older)
# ---------------------------------------------------------------------------

def binarize_age(age_series: pd.Series, threshold: int = 25) -> pd.Series:
    """Convert continuous age into two groups for parity analysis."""
    return (age_series <= threshold).astype(int).map({1: f"age<={threshold}", 0: f"age>{threshold}"})
