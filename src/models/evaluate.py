"""
Model training and evaluation pipeline – Task 4.

Orchestrates:
  1. Load engineered features from outputs/features/
  2. Prepare data (encode, split)
  3. Train Logistic Regression + Random Forest
  4. Evaluate accuracy (AUC-ROC, precision, recall)
  5. Evaluate fairness (demographic parity, equalized odds)
     on protected attributes: personal_status_sex, age
  6. Save models to outputs/models/
  7. Write JSON report to outputs/reports/evaluation_report.json

Run directly:
    python -m src.models.evaluate
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.models.credit_model import (
    MODEL_DIR,
    FEATURE_DIR,
    evaluate_model,
    prepare_data,
    save_model,
    train_models,
)
from src.models.fairness import binarize_age, fairness_report

REPORTS_DIR = "outputs/reports"


def run_evaluation() -> dict:
    """
    Full training + evaluation run.

    Returns the complete evaluation report as a dict (also saved to disk).
    """
    print("=" * 60)
    print("Task 4: Credit Risk Model Training & Fairness Evaluation")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------
    feature_path = os.path.join(FEATURE_DIR, "engineered_features.csv")
    if not os.path.exists(feature_path):
        raise FileNotFoundError(
            f"Engineered features not found at {feature_path}. "
            "Run the data pipeline first (python run_pipeline.py)."
        )

    df = pd.read_csv(feature_path)
    print(f"\n[1/5] Loaded {len(df)} samples with {df.shape[1]} columns.")

    # -----------------------------------------------------------------------
    # 2. Prepare data
    # -----------------------------------------------------------------------
    X, y, encoders, feature_names = prepare_data(df)

    # Preserve raw protected columns *before* encoding for fairness audit
    # (prepare_data has already encoded them in X, but we need the originals)
    raw_sex = df["personal_status_sex"].copy() if "personal_status_sex" in df.columns else None
    raw_age = df["age"].copy() if "age" in df.columns else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Align protected columns with the test split
    test_idx = X_test.index
    sex_test = raw_sex.loc[test_idx] if raw_sex is not None else None
    age_test = binarize_age(raw_age.loc[test_idx]) if raw_age is not None else None

    print(
        f"[2/5] Train size: {len(X_train)}  |  Test size: {len(X_test)}"
        f"  |  Default rate: {y.mean():.1%}"
    )

    # -----------------------------------------------------------------------
    # 3. Train models
    # -----------------------------------------------------------------------
    print("[3/5] Training Logistic Regression and Random Forest …")
    lr, rf, scaler = train_models(X_train, y_train)

    save_model(lr, "logistic_regression.pkl")
    save_model(rf, "random_forest.pkl")
    save_model(scaler, "scaler.pkl")
    save_model(encoders, "encoders.pkl")
    print(f"      Models saved to {MODEL_DIR}/")

    # -----------------------------------------------------------------------
    # 4. Accuracy evaluation
    # -----------------------------------------------------------------------
    print("[4/5] Evaluating accuracy …")

    lr_metrics, lr_pred, _ = evaluate_model(lr, X_test, y_test, "Logistic Regression", scaler=scaler)
    rf_metrics, rf_pred, _ = evaluate_model(rf, X_test, y_test, "Random Forest")

    for m in [lr_metrics, rf_metrics]:
        print(
            f"\n  {m['model']}"
            f"\n    AUC-ROC   : {m['auc_roc']}"
            f"\n    Precision : {m['precision']}"
            f"\n    Recall    : {m['recall']}"
        )
        print(f"\n  Classification Report:\n{m['classification_report']}")

    # -----------------------------------------------------------------------
    # 5. Fairness evaluation
    # -----------------------------------------------------------------------
    print("[5/5] Evaluating fairness …")

    fairness_results: dict = {}

    # Re-attach raw protected cols to X_test for fairness_report
    X_test_audit = X_test.copy()

    # personal_status_sex
    if sex_test is not None:
        X_test_audit["personal_status_sex_raw"] = sex_test.values
        for model_name, preds in [("Logistic Regression", lr_pred), ("Random Forest", rf_pred)]:
            rpt = fairness_report(
                y_true=y_test.values,
                y_pred=preds,
                X_test=X_test_audit,
                protected_col="personal_status_sex_raw",
                model_name=model_name,
            )
            print(f"\n{rpt['summary']}")
            fairness_results[f"{model_name}_sex"] = {
                k: v for k, v in rpt.items() if k != "summary"
            }

    # age group
    if age_test is not None:
        X_test_audit["age_group_binary"] = age_test.values
        for model_name, preds in [("Logistic Regression", lr_pred), ("Random Forest", rf_pred)]:
            rpt = fairness_report(
                y_true=y_test.values,
                y_pred=preds,
                X_test=X_test_audit,
                protected_col="age_group_binary",
                model_name=model_name,
            )
            print(f"\n{rpt['summary']}")
            fairness_results[f"{model_name}_age"] = {
                k: v for k, v in rpt.items() if k != "summary"
            }

    # -----------------------------------------------------------------------
    # 6. Save full report
    # -----------------------------------------------------------------------
    report = {
        "accuracy": {
            "logistic_regression": {k: v for k, v in lr_metrics.items() if k != "classification_report"},
            "random_forest": {k: v for k, v in rf_metrics.items() if k != "classification_report"},
        },
        "fairness": fairness_results,
        "dataset_info": {
            "total_samples": len(df),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "default_rate": round(float(y.mean()), 4),
            "features": feature_names,
        },
    }

    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n✅ Evaluation complete. Report saved to {report_path}")
    return report


if __name__ == "__main__":
    run_evaluation()

# =========================
# MULTI-DATASET EXTENSION
# =========================

import pandas as pd
from src.models.credit_model import prepare_data, train_models, evaluate_model
from sklearn.model_selection import train_test_split

print("\n--- MULTI-DATASET COMPARISON ---")

# ---------- German Dataset ----------
german = pd.read_csv("data/processed/german_credit_clean.csv")

X, y, _, _ = prepare_data(german)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr, rf, scaler = train_models(X_train, y_train)

g_lr_metrics, _, _ = evaluate_model(lr, X_test, y_test, "LR-German", scaler)
g_rf_metrics, _, _ = evaluate_model(rf, X_test, y_test, "RF-German")

print("\nGerman Dataset Results:")
print(g_lr_metrics)
print(g_rf_metrics)


# ---------- Lending Club Dataset ----------
lending = pd.read_csv("data/processed/lendingclub_clean.csv")

# ⚠️ IMPORTANT: adjust target column
if "loan_status" in lending.columns:
    lending["target"] = lending["loan_status"]
elif "default" in lending.columns:
    lending["target"] = lending["default"]
else:
    lending["target"] = lending.iloc[:, -1]

# Convert categorical
lending = pd.get_dummies(lending, drop_first=True)

X_l = lending.drop(columns=["target"])
y_l = lending["target"]

X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
    X_l, y_l, test_size=0.2, random_state=42
)

lr_l, rf_l, scaler_l = train_models(X_train_l, y_train_l)

l_lr_metrics, _, _ = evaluate_model(lr_l, X_test_l, y_test_l, "LR-Lending", scaler_l)
l_rf_metrics, _, _ = evaluate_model(rf_l, X_test_l, y_test_l, "RF-Lending")

print("\nLending Club Dataset Results:")
print(l_lr_metrics)
print(l_rf_metrics)
