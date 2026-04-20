"""
Explainability Validation — SHAP rankings vs. domain knowledge.

Goal
----
Confirm that the top features selected by SHAP are consistent with
credit-risk domain expectations. Regulators reject a model whose top
drivers are nonsensical — even if accuracy is high.

Domain priors (from CFPB, SR 11-7, FICO scorecards, German Credit
UCI description):

    Expected top-3 drivers of default risk
    ─────────────────────────────────────
    1. credit_history             – prior repayment behaviour
    2. status                     – checking-account status (proxy for income stability)
    3. duration                   – loan tenure
    4. amount / credit_amount_per_duration – loan size relative to tenure
    5. savings                    – liquid buffer
    6. employment_duration        – income stability

German Credit doesn't have a direct `income` column; `status` and
`savings` are the closest regulator-accepted proxies. Our engineered
feature `credit_amount_per_duration` captures the loan-size-per-month
concept that in real data would be "DTI ratio".

Validation rule
---------------
At least **3 of the top-5 SHAP features** for each model must come from
the domain-expected set below.

Usage
-----
    python -m src.validation.explainability_validation
    python -m src.validation.explainability_validation --min-overlap 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

from src.models.credit_model import FEATURE_DIR, load_model, prepare_data

REPORT_DIR = "outputs/reports/validation"

# Domain-expected drivers (see header note).
EXPECTED_DOMAIN_FEATURES: set[str] = {
    "credit_history",
    "status",
    "duration",
    "amount",
    "credit_amount_per_duration",   # DTI proxy
    "credit_per_person",             # DTI proxy #2
    "savings",
    "employment_duration",
    "installment_rate",
}

DEFAULT_MIN_OVERLAP = 3
TOP_N = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _top_n(shap_vals: np.ndarray, feature_names: list[str], n: int) -> list[dict]:
    mean_abs = np.abs(shap_vals).mean(axis=0)
    idx = np.argsort(mean_abs)[::-1][:n]
    return [
        {"feature": feature_names[i],
         "mean_abs_shap": round(float(mean_abs[i]), 4)}
        for i in idx
    ]


def _normalise_rf_shap(raw) -> np.ndarray:
    """Handle the 3 shap-library return shapes for binary TreeExplainer."""
    if isinstance(raw, list):
        return np.asarray(raw[1])
    arr = np.asarray(raw)
    if arr.ndim == 3:
        return arr[:, :, 1]
    return arr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(min_overlap: int = DEFAULT_MIN_OVERLAP, verbose: bool = True) -> dict:
    if verbose:
        print("=" * 60)
        print(" Explainability Validation — SHAP vs Domain Knowledge")
        print("=" * 60)

    # 1. Load models + recreate split
    lr     = load_model("logistic_regression.pkl")
    rf     = load_model("random_forest.pkl")
    scaler = load_model("scaler.pkl")

    feature_path = os.path.join(FEATURE_DIR, "engineered_features.csv")
    df = pd.read_csv(feature_path)
    X, y, _, feature_names = prepare_data(df)
    X_train, X_test, _, _ = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y,
    )
    X_train_reset = X_train.reset_index(drop=True)
    X_test_reset  = X_test.reset_index(drop=True)
    X_train_scaled = scaler.transform(X_train_reset)
    X_test_scaled  = scaler.transform(X_test_reset)

    # 2. Compute SHAP
    if verbose:
        print(f"  Computing SHAP over {len(X_test_reset)} test samples …")

    rf_exp = shap.TreeExplainer(rf)
    rf_shap = _normalise_rf_shap(rf_exp.shap_values(X_test_reset))

    lr_exp = shap.LinearExplainer(lr, X_train_scaled)
    lr_shap = lr_exp.shap_values(X_test_scaled)
    if isinstance(lr_shap, list):
        lr_shap = lr_shap[1]
    lr_shap = np.asarray(lr_shap)

    # 3. Rank top-N
    rf_top = _top_n(rf_shap, feature_names, TOP_N)
    lr_top = _top_n(lr_shap, feature_names, TOP_N)

    def _overlap(top_list: list[dict]) -> tuple[int, list[str], list[str]]:
        top_names   = [f["feature"] for f in top_list]
        hits        = [f for f in top_names if f in EXPECTED_DOMAIN_FEATURES]
        unexpected  = [f for f in top_names if f not in EXPECTED_DOMAIN_FEATURES]
        return len(hits), hits, unexpected

    rf_hits, rf_matched, rf_unexpected = _overlap(rf_top)
    lr_hits, lr_matched, lr_unexpected = _overlap(lr_top)

    rf_pass = rf_hits >= min_overlap
    lr_pass = lr_hits >= min_overlap

    result = {
        "expected_domain_features": sorted(EXPECTED_DOMAIN_FEATURES),
        "top_n":            TOP_N,
        "min_overlap":      min_overlap,
        "random_forest": {
            "top_features":         rf_top,
            "matched_domain":       rf_matched,
            "unexpected":           rf_unexpected,
            "domain_overlap_count": rf_hits,
            "passed":               rf_pass,
        },
        "logistic_regression": {
            "top_features":         lr_top,
            "matched_domain":       lr_matched,
            "unexpected":           lr_unexpected,
            "domain_overlap_count": lr_hits,
            "passed":               lr_pass,
        },
        "overall_passed": rf_pass and lr_pass,
    }

    # 4. Save + print
    os.makedirs(REPORT_DIR, exist_ok=True)
    out_path = os.path.join(REPORT_DIR, "explainability_validation.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    if verbose:
        for label, block in [("Random Forest",       result["random_forest"]),
                             ("Logistic Regression", result["logistic_regression"])]:
            print(f"\n  {label}")
            print(f"  Top-{TOP_N} features by mean |SHAP|:")
            for i, f in enumerate(block["top_features"], 1):
                marker = "✓" if f["feature"] in EXPECTED_DOMAIN_FEATURES else "○"
                print(f"    {i}. [{marker}] {f['feature']:<32} {f['mean_abs_shap']:.4f}")
            print(f"    Domain matches: {block['domain_overlap_count']}/{TOP_N}  "
                  f"(need ≥{min_overlap})   "
                  f"{'PASS' if block['passed'] else 'FAIL'}")

        print(f"\n  Report → {out_path}")
        print(f"  Overall: {'PASS' if result['overall_passed'] else 'FAIL'}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    p = argparse.ArgumentParser(description="Explainability validation")
    p.add_argument("--min-overlap", type=int, default=DEFAULT_MIN_OVERLAP,
                   help="minimum number of top-N features that must be in "
                        "the domain-expected set")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    res = run(min_overlap=args.min_overlap, verbose=not args.quiet)
    sys.exit(0 if res["overall_passed"] else 1)


if __name__ == "__main__":
    _cli()
