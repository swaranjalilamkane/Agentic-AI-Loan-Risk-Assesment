"""
Model training and evaluation pipeline - Task 4 + Task 5.

Steps
-----
1. Load engineered features
2. Prepare data (encode, split)
3. Train Logistic Regression + Random Forest
4. Evaluate accuracy (AUC-ROC, precision, recall)
5. Find and save per-group fairness thresholds (bias mitigation)
6. Evaluate fairness using bias-mitigated predictions as the default
7. Save models, thresholds, and evaluation report

All predictions in production go through predict_fair() — the
group-specific threshold is always applied.

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
from src.models.bias_mitigation import find_and_save_thresholds, predict_fair
from src.data_integration.augment_data import augment_training_data

REPORTS_DIR = "outputs/reports"


def run_evaluation() -> dict:
    """
    Full training + fairness-aware evaluation run.
    Returns the complete evaluation report (also saved to disk).
    """
    print("=" * 60)
    print("Task 4 + 5: Credit Risk Model Training & Fair Evaluation")
    print("=" * 60)

    # ── 1. Load data ─────────────────────────────────────────────────────────
    feature_path = os.path.join(FEATURE_DIR, "engineered_features.csv")
    if not os.path.exists(feature_path):
        raise FileNotFoundError(
            f"Engineered features not found at {feature_path}. "
            "Run the data pipeline first (python run_pipeline.py)."
        )

    df = pd.read_csv(feature_path)
    print(f"\n[1/6] Loaded {len(df)} samples with {df.shape[1]} columns.")

    # ── 2. Prepare data ───────────────────────────────────────────────────────
    X, y, encoders, feature_names = prepare_data(df)

    # Preserve raw protected columns before encoding for fairness audit
    raw_sex = df["personal_status_sex"].copy() if "personal_status_sex" in df.columns else None
    raw_age = df["age"].copy()                  if "age"                  in df.columns else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    test_idx  = X_test.index
    sex_test  = raw_sex.loc[test_idx].reset_index(drop=True).values  if raw_sex is not None else None
    age_test  = binarize_age(raw_age.loc[test_idx]).reset_index(drop=True).values if raw_age is not None else None

    print(
        f"[2/6] Train size: {len(X_train)}  |  Test size: {len(X_test)}"
        f"  |  Default rate: {y.mean():.1%}"
    )

    # ── 2b. Augment training data for small protected-attribute groups ────────
    # Rebuilds the training rows with the original (pre-encoding) columns so
    # the interpolation uses human-readable categorical values, then re-encodes.
    train_idx      = X_train.index
    df_train_raw   = df.loc[train_idx].copy().reset_index(drop=True)
    df_train_aug   = augment_training_data(
        df_train_raw,
        group_col="personal_status_sex",
        min_group_size=150,
        random_state=42,
    )
    print(
        f"      Augmented train: {len(df_train_raw)} → {len(df_train_aug)} rows"
        f"  (+{len(df_train_aug) - len(df_train_raw)} synthetic)"
    )
    X_train_aug, y_train_aug, _, _ = prepare_data(df_train_aug)

    # ── 3. Train models ───────────────────────────────────────────────────────
    print("[3/6] Training Logistic Regression and Random Forest …")
    lr, rf, scaler = train_models(X_train_aug, y_train_aug)

    save_model(lr,       "logistic_regression.pkl")
    save_model(rf,       "random_forest.pkl")
    save_model(scaler,   "scaler.pkl")
    save_model(encoders, "encoders.pkl")
    print(f"      Models saved to {MODEL_DIR}/")

    # ── 4. Accuracy evaluation ────────────────────────────────────────────────
    print("[4/6] Evaluating accuracy …")

    lr_metrics, _, _ = evaluate_model(lr, X_test, y_test, "Logistic Regression", scaler=scaler)
    rf_metrics, _, _ = evaluate_model(rf, X_test, y_test, "Random Forest")

    for m in [lr_metrics, rf_metrics]:
        print(
            f"\n  {m['model']}"
            f"\n    AUC-ROC   : {m['auc_roc']}"
            f"\n    Precision : {m['precision']}"
            f"\n    Recall    : {m['recall']}"
        )
        print(f"\n  Classification Report:\n{m['classification_report']}")

    # ── 5. Find and save fairness thresholds ─────────────────────────────────
    print("[5/6] Finding and saving per-group fairness thresholds …")
    mitigation_report = find_and_save_thresholds(
        lr=lr, rf=rf, scaler=scaler,
        X_test=X_test, y_test=y_test,
        sex_test=sex_test, age_test_bin=age_test,
    )
    print("      Thresholds saved to outputs/models/fairness_thresholds.json")

    # ── 6. Fairness evaluation using fair predictions ─────────────────────────
    print("[6/6] Evaluating fairness (bias-mitigated predictions) …")

    fairness_results: dict = {}
    X_test_audit = X_test.copy()

    # personal_status_sex
    if sex_test is not None:
        X_test_audit["personal_status_sex_raw"] = sex_test

        lr_pred_fair = predict_fair(
            lr, X_test, sex_test,
            model_name="logistic_regression",
            protected_attr="personal_status_sex",
            scaler=scaler,
        )
        rf_pred_fair = predict_fair(
            rf, X_test, sex_test,
            model_name="random_forest",
            protected_attr="personal_status_sex",
        )

        for model_name, preds in [
            ("Logistic Regression", lr_pred_fair),
            ("Random Forest",       rf_pred_fair),
        ]:
            rpt = fairness_report(
                y_true=y_test.values, y_pred=preds,
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
        X_test_audit["age_group_binary"] = age_test

        lr_pred_fair_age = predict_fair(
            lr, X_test, age_test,
            model_name="logistic_regression",
            protected_attr="age_group_binary",
            scaler=scaler,
        )
        rf_pred_fair_age = predict_fair(
            rf, X_test, age_test,
            model_name="random_forest",
            protected_attr="age_group_binary",
        )

        for model_name, preds in [
            ("Logistic Regression", lr_pred_fair_age),
            ("Random Forest",       rf_pred_fair_age),
        ]:
            rpt = fairness_report(
                y_true=y_test.values, y_pred=preds,
                X_test=X_test_audit,
                protected_col="age_group_binary",
                model_name=model_name,
            )
            print(f"\n{rpt['summary']}")
            fairness_results[f"{model_name}_age"] = {
                k: v for k, v in rpt.items() if k != "summary"
            }

    # ── 7. Save full report ───────────────────────────────────────────────────
    report = {
        "accuracy": {
            "logistic_regression": {k: v for k, v in lr_metrics.items() if k != "classification_report"},
            "random_forest":       {k: v for k, v in rf_metrics.items() if k != "classification_report"},
        },
        "fairness":       fairness_results,
        "bias_mitigation": mitigation_report,
        "dataset_info": {
            "total_samples":  len(df),
            "train_samples":  len(X_train),
            "test_samples":   len(X_test),
            "default_rate":   round(float(y.mean()), 4),
            "features":       feature_names,
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
