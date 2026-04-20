"""
Fairness Validation — German Credit.

Goal
----
Confirm the trained Logistic Regression and Random Forest satisfy fairness
constraints across three protected attributes:

    * gender (binarised from personal_status_sex)
    * age    (binarised at 25)
    * foreign_worker

Methodology
-----------
1. Recreate the exact train/test split used in training (random_state=42).
2. For each model × protected attribute, compute:
       - Demographic Parity Difference (DPD)
       - Equalized Odds Difference      (EOD)
       via src.models.fairness.fairness_report()
3. Flag PASS when DPD ≤ 0.10 AND EOD ≤ 0.10 (standard regulatory line).
4. Emit JSON report + printable summary table.

Usage
-----
    python -m src.validation.fairness_validation
    python -m src.validation.fairness_validation --threshold 0.05   # stricter
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
from sklearn.model_selection import train_test_split

from src.models.credit_model import FEATURE_DIR, load_model, prepare_data
from src.models.fairness import binarize_age, fairness_report

REPORT_DIR = "outputs/reports/validation"
DEFAULT_BIAS_THRESHOLD = 0.10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _binarise_gender(sex_series: pd.Series) -> pd.Series:
    """
    German Credit 'personal_status_sex' codes:
       A91 = male / divorced or separated
       A92 = female
       A93 = male / single
       A94 = male / married or widowed
    """
    return sex_series.astype(str).apply(
        lambda c: "female" if c.upper() == "A92" else "male"
    )


def _evaluate(
    model, X_test, y_test, model_name: str, scaler=None,
) -> tuple[np.ndarray, np.ndarray]:
    X_eval = scaler.transform(X_test) if scaler is not None else X_test
    y_pred = model.predict(X_eval)
    return np.array(y_pred), np.array(y_test)


def _check(report: dict, threshold: float) -> bool:
    if "error" in report:
        return False
    return (
        report["demographic_parity_difference"] <= threshold
        and report["equalized_odds_difference"]   <= threshold
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(threshold: float = DEFAULT_BIAS_THRESHOLD, verbose: bool = True) -> dict:
    if verbose:
        print("=" * 60)
        print(f" Fairness Validation — German Credit (threshold={threshold})")
        print("=" * 60)

    # 1. Load artifacts
    lr     = load_model("logistic_regression.pkl")
    rf     = load_model("random_forest.pkl")
    scaler = load_model("scaler.pkl")

    feature_path = os.path.join(FEATURE_DIR, "engineered_features.csv")
    df = pd.read_csv(feature_path)

    # Preserve raw protected columns BEFORE encoding
    raw_sex = df.get("personal_status_sex")
    raw_age = df.get("age")
    raw_foreign = df.get("foreign_worker")

    X, y, _, _ = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y,
    )
    test_idx = X_test.index

    # Build an auxiliary dataframe with human-readable protected columns
    X_test_aux = X_test.copy()
    if raw_sex is not None:
        X_test_aux["gender_bin"] = _binarise_gender(raw_sex.loc[test_idx]).values
    if raw_age is not None:
        X_test_aux["age_bin"] = binarize_age(raw_age.loc[test_idx]).values
    if raw_foreign is not None:
        # A201 = yes foreign, A202 = no foreign (UCI codes)
        X_test_aux["foreign_worker_bin"] = (
            raw_foreign.loc[test_idx]
            .astype(str)
            .apply(lambda c: "foreign" if c.upper() == "A201" else "domestic")
            .values
        )

    # 2. Predictions
    rf_pred, y_true = _evaluate(rf, X_test, y_test, "Random Forest")
    lr_pred, _      = _evaluate(lr, X_test, y_test, "Logistic Regression", scaler=scaler)

    # Note: in this pipeline positive class = 1 = "default/reject".
    # For fairness we want approval parity, so flip to approval (=0):
    rf_approved = (rf_pred == 0).astype(int)
    lr_approved = (lr_pred == 0).astype(int)
    y_approved  = (y_true == 0).astype(int)

    # 3. Run reports
    protected_cols = [c for c in
                      ("gender_bin", "age_bin", "foreign_worker_bin")
                      if c in X_test_aux.columns]

    results: dict = {
        "threshold": threshold,
        "models": {"random_forest": {}, "logistic_regression": {}},
    }

    for col in protected_cols:
        rf_rep = fairness_report(
            y_approved, rf_approved, X_test_aux, col, model_name="Random Forest")
        lr_rep = fairness_report(
            y_approved, lr_approved, X_test_aux, col, model_name="Logistic Regression")

        rf_rep["passed"] = _check(rf_rep, threshold)
        lr_rep["passed"] = _check(lr_rep, threshold)

        results["models"]["random_forest"][col]        = rf_rep
        results["models"]["logistic_regression"][col]  = lr_rep

    # 4. Overall pass flag
    def _all_pass(model_reports: dict) -> bool:
        return all(r.get("passed", False) for r in model_reports.values())

    rf_ok = _all_pass(results["models"]["random_forest"])
    lr_ok = _all_pass(results["models"]["logistic_regression"])
    results["random_forest_passed"] = rf_ok
    results["logistic_regression_passed"] = lr_ok
    results["overall_passed"] = rf_ok and lr_ok

    # 5. Save
    os.makedirs(REPORT_DIR, exist_ok=True)
    out_path = os.path.join(REPORT_DIR, "fairness_validation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # 6. Print
    if verbose:
        for model_label, key in [("Random Forest", "random_forest"),
                                 ("Logistic Regression", "logistic_regression")]:
            print(f"\n  {model_label}")
            print(f"  {'Attribute':<22}{'DPD':>8}{'EOD':>8}   Status")
            print("  " + "-" * 50)
            for col, rep in results["models"][key].items():
                dpd = rep["demographic_parity_difference"]
                eod = rep["equalized_odds_difference"]
                ok  = "PASS" if rep["passed"] else "FAIL"
                print(f"  {col:<22}{dpd:>8.4f}{eod:>8.4f}   {ok}")

        print(f"\n  Report → {out_path}")
        print(f"  Overall: {'PASS' if results['overall_passed'] else 'FAIL'}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    p = argparse.ArgumentParser(description="Fairness validation on German Credit")
    p.add_argument("--threshold", type=float, default=DEFAULT_BIAS_THRESHOLD)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    res = run(threshold=args.threshold, verbose=not args.quiet)
    sys.exit(0 if res["overall_passed"] else 1)


if __name__ == "__main__":
    _cli()
