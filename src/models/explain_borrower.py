"""
Borrower Explanation Module – Task 5 (SHAP → Natural Language).

Converts SHAP values for a single borrower into a plain-English explanation
of why their application was flagged as high or low risk.

Usage
-----
    # Explain all test-set borrowers (saves outputs/reports/borrower_explanations.json)
    from src.models.explain_borrower import explain_all_borrowers
    explanations = explain_all_borrowers()

    # Explain a single borrower by test-set index
    from src.models.explain_borrower import explain_single_borrower
    result = explain_single_borrower(borrower_index=5, model="rf")
    print(result["narrative"])

    # Standalone CLI
    python -m src.models.explain_borrower
"""

from __future__ import annotations

import json
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

from src.models.credit_model import (
    FEATURE_DIR,
    load_model,
    prepare_data,
)
from src.utils.logger import logger

REPORT_DIR = "outputs/reports"


# ---------------------------------------------------------------------------
# Feature → Human-readable phrase
# ---------------------------------------------------------------------------

# Maps known raw categorical values to concise English.
_CATEGORY_LABELS: dict[str, dict[str, str]] = {
    "status": {
        "... < 100 DM":                                         "checking balance < 100 DM",
        "0 <= ... < 200 DM":                                    "checking balance 100–200 DM",
        "... >= 200 DM / salary assignments for at least 1 year": "checking balance ≥ 200 DM",
        "no checking account":                                  "no checking account",
    },
    "credit_history": {
        "no credits taken/all credits paid back duly":          "no previous credits (or all repaid)",
        "all credits at this bank paid back duly":              "all bank credits repaid on time",
        "existing credits paid back duly till now":             "existing credits paid on time so far",
        "delay in paying off in the past":                      "past payment delays",
        "critical account/other credits existing":              "critical account or other active credits",
    },
    "purpose": {
        "car (new)":                    "new car purchase",
        "car (used)":                   "used car purchase",
        "furniture/equipment":          "furniture / equipment",
        "radio/television":             "radio / television",
        "domestic appliances":          "domestic appliances",
        "repairs":                      "repairs",
        "education":                    "education",
        "retraining":                   "retraining",
        "business":                     "business",
        "others":                       "other purpose",
    },
    "savings": {
        "... < 100 DM":                 "savings < 100 DM",
        "100 <= ... < 500 DM":          "savings 100–500 DM",
        "500 <= ... < 1000 DM":         "savings 500–1000 DM",
        "... >= 1000 DM":               "savings ≥ 1000 DM",
        "unknown/no savings account":   "no savings account",
    },
    "employment_duration": {
        "unemployed":                   "currently unemployed",
        "... < 1 year":                 "employed < 1 year",
        "1 <= ... < 4 years":           "employed 1–4 years",
        "4 <= ... < 7 years":           "employed 4–7 years",
        "... >= 7 years":               "employed ≥ 7 years",
    },
    "personal_status_sex": {
        "male : divorced/separated":            "male, divorced/separated",
        "female : divorced/separated/married":  "female",
        "male : single":                        "male, single",
        "male : married/widowed":               "male, married/widowed",
    },
    "other_debtors": {
        "none":          "no co-applicant or guarantor",
        "co-applicant":  "has a co-applicant",
        "guarantor":     "has a guarantor",
    },
    "property": {
        "real estate":                                                "real estate ownership",
        "building society savings agreement/life insurance":          "life insurance / savings plan",
        "car or other, not in attribute 6":                          "car or other asset",
        "unknown / no property":                                     "no known property",
    },
    "other_installment_plans": {
        "bank":    "other bank instalment plan",
        "stores":  "store instalment plan",
        "none":    "no other instalment plans",
    },
    "housing": {
        "rent":     "renting",
        "own":      "owns home",
        "for free": "lives rent-free",
    },
    "job": {
        "unemployed/ unskilled - non-resident": "unemployed or unskilled non-resident",
        "unskilled - resident":                 "unskilled resident",
        "skilled employee / official":          "skilled employee",
        "management/ self-employed/highly qualified employee/ officer":
                                                "management / highly qualified",
    },
    "telephone": {
        "none":                         "no telephone registered",
        "yes, registered under the customer's name": "telephone registered",
    },
    "foreign_worker": {
        "yes": "foreign worker",
        "no":  "domestic worker",
    },
}

# Human-readable labels for every feature (used when building sentences).
_FEATURE_LABELS: dict[str, str] = {
    "status":                   "Checking account status",
    "duration":                 "Loan duration",
    "credit_history":           "Credit history",
    "purpose":                  "Loan purpose",
    "amount":                   "Credit amount",
    "savings":                  "Savings account",
    "employment_duration":      "Employment length",
    "installment_rate":         "Instalment rate",
    "personal_status_sex":      "Personal status & sex",
    "other_debtors":            "Co-applicant / guarantor",
    "present_residence":        "Years at current address",
    "property":                 "Property / assets",
    "age":                      "Applicant age",
    "other_installment_plans":  "Other instalment plans",
    "housing":                  "Housing situation",
    "number_credits":           "Existing credits at this bank",
    "job":                      "Job type",
    "people_liable":            "Number of dependants",
    "telephone":                "Telephone",
    "foreign_worker":           "Residency status",
    "credit_amount_per_duration": "Monthly repayment burden",
    "credit_per_person":        "Credit per household member",
}


def _describe_value(feature: str, raw_value) -> str:
    """Return a human-readable description of a single feature value."""
    if feature in _CATEGORY_LABELS:
        key = str(raw_value).strip()
        return _CATEGORY_LABELS[feature].get(key, key)

    # Numeric features — add units/context
    try:
        v = float(raw_value)
    except (ValueError, TypeError):
        return str(raw_value)

    if feature == "duration":
        return f"{int(v)} months"
    if feature == "amount":
        return f"{int(v):,} DM"
    if feature == "age":
        return f"{int(v)} years old"
    if feature == "installment_rate":
        return f"{int(v)}% of disposable income"
    if feature == "present_residence":
        return f"{int(v)} year(s) at current address"
    if feature == "number_credits":
        return f"{int(v)} existing credit(s)"
    if feature == "people_liable":
        return f"{int(v)} dependent(s)"
    if feature == "credit_amount_per_duration":
        return f"{v:.0f} DM/month repayment burden"
    if feature == "credit_per_person":
        return f"{v:,.0f} DM per household member"

    return str(raw_value)


# ---------------------------------------------------------------------------
# Sentence builder
# ---------------------------------------------------------------------------

def _build_sentence(feature: str, raw_value, shap_val: float) -> dict:
    """
    Build a single-factor explanation entry.

    Returns a dict with:
        factor      – human label of the feature
        value       – human description of the feature value
        direction   – "increases risk" | "reduces risk"
        shap        – raw SHAP value (float, rounded)
        sentence    – full readable sentence
    """
    label     = _FEATURE_LABELS.get(feature, feature.replace("_", " ").title())
    value_str = _describe_value(feature, raw_value)
    direction = "increases default risk" if shap_val > 0 else "reduces default risk"
    magnitude = abs(shap_val)

    # Magnitude qualifier
    if magnitude >= 0.30:
        strength = "strongly"
    elif magnitude >= 0.15:
        strength = "moderately"
    elif magnitude >= 0.05:
        strength = "slightly"
    else:
        strength = "marginally"

    sentence = f"{label} ({value_str}) {strength} {direction}."

    return {
        "feature":   feature,
        "factor":    label,
        "value":     value_str,
        "direction": direction,
        "shap":      round(float(shap_val), 4),
        "sentence":  sentence,
    }


# ---------------------------------------------------------------------------
# Core explainer
# ---------------------------------------------------------------------------

def explain_borrower(
    *,
    shap_values: np.ndarray,
    raw_row: pd.Series,
    feature_names: list[str],
    base_value: float,
    predicted_prob: float,
    predicted_label: int,
    actual_label: int | None = None,
    borrower_id: int | None = None,
    top_n: int = 5,
) -> dict:
    """
    Build a structured explanation for one borrower.

    Parameters
    ----------
    shap_values     : 1D array of SHAP values (one per feature, class=1/default)
    raw_row         : pd.Series with original (pre-encoding) feature values
    feature_names   : list of feature names matching shap_values
    base_value      : SHAP base value (model's average log-odds)
    predicted_prob  : float, P(default) for this borrower
    predicted_label : 0 (approved) or 1 (rejected as high risk)
    actual_label    : ground truth if available
    borrower_id     : optional integer index for tracking
    top_n           : how many risk / protective factors to list

    Returns
    -------
    dict with keys: borrower_id, decision, probability, risk_level,
                    risk_factors, protective_factors, narrative, factors_detail
    """
    assert len(shap_values) == len(feature_names), "SHAP length mismatch"

    # --- Build factor list ---------------------------------------------------
    factors = []
    for i, (feat, sv) in enumerate(zip(feature_names, shap_values)):
        raw_val = raw_row.get(feat, "N/A") if hasattr(raw_row, "get") else "N/A"
        factors.append(_build_sentence(feat, raw_val, sv))

    # Split by direction, rank by |SHAP|
    risk_factors       = sorted(
        [f for f in factors if f["shap"] > 0],
        key=lambda x: x["shap"], reverse=True
    )[:top_n]
    protective_factors = sorted(
        [f for f in factors if f["shap"] < 0],
        key=lambda x: x["shap"]          # most negative first
    )[:top_n]

    # --- Risk level ----------------------------------------------------------
    if predicted_prob >= 0.75:
        risk_level = "Very High Risk"
    elif predicted_prob >= 0.55:
        risk_level = "High Risk"
    elif predicted_prob >= 0.40:
        risk_level = "Moderate Risk"
    elif predicted_prob >= 0.25:
        risk_level = "Low Risk"
    else:
        risk_level = "Very Low Risk"

    decision = "REJECTED (predicted default)" if predicted_label == 1 else "APPROVED (predicted good credit)"

    # --- Narrative paragraph -------------------------------------------------
    lines = [
        f"LOAN DECISION: {decision}",
        f"Default Probability: {predicted_prob:.1%}  |  Risk Level: {risk_level}",
    ]
    if actual_label is not None:
        actual_str = "default" if actual_label == 1 else "good credit"
        lines.append(f"Actual Outcome (ground truth): {actual_str}")

    lines.append("")
    lines.append("TOP REASONS THIS APPLICATION WAS FLAGGED:")
    if risk_factors:
        for i, f in enumerate(risk_factors, 1):
            lines.append(f"  {i}. {f['sentence']}")
    else:
        lines.append("  (No significant risk factors found.)")

    lines.append("")
    lines.append("FACTORS IN THE APPLICANT'S FAVOUR:")
    if protective_factors:
        for i, f in enumerate(protective_factors, 1):
            lines.append(f"  {i}. {f['sentence']}")
    else:
        lines.append("  (No significant protective factors found.)")

    lines.append("")
    lines.append(
        f"Note: This explanation is based on the model's SHAP values. "
        f"The base default rate for the population is "
        f"{1 / (1 + np.exp(-base_value)):.1%}. "
        f"Individual factors above/below this baseline drive the final score."
    )

    narrative = "\n".join(lines)

    return {
        "borrower_id":          borrower_id,
        "decision":             decision,
        "probability":          round(float(predicted_prob), 4),
        "risk_level":           risk_level,
        "actual_label":         actual_label,
        "risk_factors":         risk_factors,
        "protective_factors":   protective_factors,
        "narrative":            narrative,
        "factors_detail":       factors,   # all features, sorted by |SHAP|
    }


# ---------------------------------------------------------------------------
# Pipeline helpers — load models, compute SHAP, explain all test borrowers
# ---------------------------------------------------------------------------

def _load_pipeline_artifacts():
    """Load models, recreate exact train/test split, compute SHAP values."""
    feature_path = os.path.join(FEATURE_DIR, "engineered_features.csv")
    df = pd.read_csv(feature_path)

    raw_df = df.copy()  # keep pre-encoding values for narrative

    X, y, _, feature_names = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    test_idx = X_test.index

    lr     = load_model("logistic_regression.pkl")
    rf     = load_model("random_forest.pkl")
    scaler = load_model("scaler.pkl")

    X_test_reset  = X_test.reset_index(drop=True)
    X_train_reset = X_train.reset_index(drop=True)
    y_test_reset  = y_test.reset_index(drop=True)
    X_test_scaled = scaler.transform(X_test_reset)
    X_train_scaled = scaler.transform(X_train_reset)

    # RF SHAP
    rf_explainer = shap.TreeExplainer(rf)
    rf_shap_raw  = rf_explainer.shap_values(X_test_reset)
    if isinstance(rf_shap_raw, list):
        rf_shap_vals = rf_shap_raw[1]
    elif isinstance(rf_shap_raw, np.ndarray) and rf_shap_raw.ndim == 3:
        rf_shap_vals = rf_shap_raw[:, :, 1]
    else:
        rf_shap_vals = rf_shap_raw

    ev = rf_explainer.expected_value
    rf_base = float(np.array(ev).flatten()[1] if isinstance(ev, (list, np.ndarray)) and len(np.array(ev).flatten()) > 1 else np.array(ev).flatten()[0])

    # LR SHAP
    lr_explainer = shap.LinearExplainer(lr, X_train_scaled)
    lr_shap_vals = lr_explainer.shap_values(X_test_scaled)
    if isinstance(lr_shap_vals, list):
        lr_shap_vals = lr_shap_vals[1]

    ev_lr = lr_explainer.expected_value
    lr_base = float(np.array(ev_lr).flatten()[0])

    # Predictions
    rf_probs  = rf.predict_proba(X_test_reset.values)[:, 1]
    rf_preds  = rf.predict(X_test_reset.values)
    lr_probs  = lr.predict_proba(X_test_scaled)[:, 1]
    lr_preds  = lr.predict(X_test_scaled)

    raw_test = raw_df.loc[test_idx].reset_index(drop=True)

    return {
        "feature_names": feature_names,
        "y_test":        y_test_reset,
        "raw_test":      raw_test,
        "rf": {
            "shap_vals":  rf_shap_vals,
            "base_value": rf_base,
            "probs":      rf_probs,
            "preds":      rf_preds,
        },
        "lr": {
            "shap_vals":  lr_shap_vals,
            "base_value": lr_base,
            "probs":      lr_probs,
            "preds":      lr_preds,
        },
    }


def explain_single_borrower(
    borrower_index: int = 0,
    model: str = "rf",
    print_narrative: bool = True,
) -> dict:
    """
    Explain one borrower from the test set.

    Parameters
    ----------
    borrower_index : position in the test set (0-based)
    model          : "rf" (Random Forest) or "lr" (Logistic Regression)
    print_narrative: if True, print the narrative to stdout

    Returns
    -------
    Explanation dict (see explain_borrower return signature)
    """
    artifacts = _load_pipeline_artifacts()
    m = artifacts[model]
    fn = artifacts["feature_names"]
    raw_test = artifacts["raw_test"]
    y_test   = artifacts["y_test"]

    if borrower_index >= len(raw_test):
        raise IndexError(
            f"borrower_index={borrower_index} out of range "
            f"(test set has {len(raw_test)} samples)"
        )

    result = explain_borrower(
        shap_values     = m["shap_vals"][borrower_index],
        raw_row         = raw_test.iloc[borrower_index],
        feature_names   = fn,
        base_value      = m["base_value"],
        predicted_prob  = m["probs"][borrower_index],
        predicted_label = int(m["preds"][borrower_index]),
        actual_label    = int(y_test.iloc[borrower_index]),
        borrower_id     = borrower_index,
    )

    if print_narrative:
        print("\n" + "=" * 60)
        print(f"BORROWER #{borrower_index} — {'Random Forest' if model == 'rf' else 'Logistic Regression'}")
        print("=" * 60)
        print(result["narrative"])

    return result


def explain_all_borrowers(
    model: str = "rf",
    save_json: bool = True,
) -> list[dict]:
    """
    Generate explanations for every borrower in the test set.

    Parameters
    ----------
    model     : "rf" or "lr"
    save_json : save to outputs/reports/borrower_explanations_{model}.json

    Returns
    -------
    List of explanation dicts (one per test borrower)
    """
    print(f"\n{'='*60}")
    print(f"Generating SHAP explanations for all test borrowers ({model.upper()})")
    print("=" * 60)

    artifacts = _load_pipeline_artifacts()
    m  = artifacts[model]
    fn = artifacts["feature_names"]
    raw_test = artifacts["raw_test"]
    y_test   = artifacts["y_test"]
    n        = len(raw_test)

    explanations = []
    for i in range(n):
        exp = explain_borrower(
            shap_values     = m["shap_vals"][i],
            raw_row         = raw_test.iloc[i],
            feature_names   = fn,
            base_value      = m["base_value"],
            predicted_prob  = m["probs"][i],
            predicted_label = int(m["preds"][i]),
            actual_label    = int(y_test.iloc[i]),
            borrower_id     = i,
        )
        explanations.append(exp)

    # Summary stats
    n_rejected = sum(1 for e in explanations if e["decision"].startswith("REJECTED"))
    n_approved  = n - n_rejected

    print(f"\n  Test set: {n} borrowers")
    print(f"  Approved : {n_approved} ({n_approved/n:.1%})")
    print(f"  Rejected : {n_rejected} ({n_rejected/n:.1%})")

    # Print a few sample narratives
    high_risk = [e for e in explanations if e["risk_level"] in ("Very High Risk", "High Risk")]
    if high_risk:
        print(f"\n--- Sample: Highest-risk borrower (#{high_risk[0]['borrower_id']}) ---")
        print(high_risk[0]["narrative"])

    if save_json:
        os.makedirs(REPORT_DIR, exist_ok=True)
        path = os.path.join(REPORT_DIR, f"borrower_explanations_{model}.json")
        # Strip 'factors_detail' for smaller file; keep narratives + key fields
        slim = [
            {k: v for k, v in e.items() if k != "factors_detail"}
            for e in explanations
        ]
        with open(path, "w") as f:
            json.dump(slim, f, indent=2)
        print(f"\n  Saved → {path}")

    return explanations


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    model = "rf"
    index = None

    for a in args:
        if a in ("--lr", "--logistic"):
            model = "lr"
        elif a.isdigit():
            index = int(a)

    if index is not None:
        explain_single_borrower(borrower_index=index, model=model)
    else:
        explain_all_borrowers(model=model)
