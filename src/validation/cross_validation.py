"""
Cross-Validation — K-fold on the Lending Club dataset.

Goal
----
Confirm that model performance (AUC, precision, recall, F1) is consistent
across K independent data partitions, so we're not overfit to any single
train/test split.

Methodology
-----------
1. Load data/processed/lendingclub_clean.csv
2. Minimal feature engineering: numeric imputation + label encoding + DTI
3. Stratified K-Fold (K=5) on both Logistic Regression and Random Forest,
   using the same hyperparameters as src/models/credit_model.py
4. Flag as PASS when std(AUC) / mean(AUC) < 0.05  (CV coefficient of variation)

Usage
-----
    python -m src.validation.cross_validation          # K=5
    python -m src.validation.cross_validation --k 10   # K=10
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

LC_PATH   = "data/processed/lendingclub_clean.csv"
REPORT_DIR = "outputs/reports/validation"
CV_STD_THRESHOLD = 0.05   # coefficient of variation on AUC


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _prepare_lending_club(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Minimal encoder matching the Lending Club schema."""
    df = df.copy()

    # Drop pipeline-artifact columns
    df = df.drop(columns=[c for c in df.columns if c.startswith("_")], errors="ignore")
    df = df.drop(columns=["loan_id"], errors="ignore")

    # Binary target: N (rejected / default-like) = 1, Y (approved) = 0
    if "loan_status" not in df.columns:
        raise ValueError("Expected 'loan_status' column in Lending Club data.")
    df["target"] = (df["loan_status"].astype(str).str.upper() == "N").astype(int)
    df = df.drop(columns=["loan_status"])

    # Numeric columns → median impute
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Engineered: debt-to-income ratio (if possible)
    if {"loanamount", "applicantincome"}.issubset(df.columns):
        income_safe = df["applicantincome"].replace(0, np.nan).fillna(
            df["applicantincome"].median()
        )
        df["dti_ratio"] = (df["loanamount"].fillna(0) / income_safe).round(4)

    # Object columns → LabelEncoder
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("MISSING")
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Final NaN sweep
    df = df.fillna(0)

    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


# ---------------------------------------------------------------------------
# Cross-validation loop
# ---------------------------------------------------------------------------

def _fold_metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "auc":       round(float(roc_auc_score(y_true, y_prob)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }


def _summarise(fold_rows: list[dict]) -> dict:
    """Mean/std across folds for each metric."""
    keys = ("auc", "precision", "recall", "f1")
    return {
        k: {
            "mean": round(float(np.mean([r[k] for r in fold_rows])), 4),
            "std":  round(float(np.std([r[k] for r in fold_rows])),  4),
            "min":  round(float(np.min([r[k] for r in fold_rows])),  4),
            "max":  round(float(np.max([r[k] for r in fold_rows])),  4),
        }
        for k in keys
    }


def run(k: int = 5, verbose: bool = True) -> dict:
    """Execute K-fold CV on Lending Club for LR + RF."""
    if verbose:
        print("=" * 60)
        print(f" Cross-Validation — Lending Club (K={k})")
        print("=" * 60)

    if not os.path.exists(LC_PATH):
        raise FileNotFoundError(f"Missing data file: {LC_PATH}")

    df = pd.read_csv(LC_PATH)
    X, y = _prepare_lending_club(df)
    if verbose:
        print(f"  Samples: {len(X)}   Features: {X.shape[1]}   "
              f"Positive rate: {y.mean():.2%}")

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    lr_rows, rf_rows = [], []
    for i, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        # --- Logistic Regression (scaled) ---
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        lr = LogisticRegression(
            class_weight="balanced", max_iter=1000,
            solver="lbfgs", random_state=42,
        ).fit(X_tr_s, y_tr)
        lr_pred = lr.predict(X_te_s)
        lr_prob = lr.predict_proba(X_te_s)[:, 1]
        lr_rows.append({"fold": i, **_fold_metrics(y_te, lr_pred, lr_prob)})

        # --- Random Forest (unscaled) ---
        rf = RandomForestClassifier(
            class_weight="balanced", n_estimators=100, max_depth=8,
            random_state=42, n_jobs=-1,
        ).fit(X_tr, y_tr)
        rf_pred = rf.predict(X_te)
        rf_prob = rf.predict_proba(X_te)[:, 1]
        rf_rows.append({"fold": i, **_fold_metrics(y_te, rf_pred, rf_prob)})

        if verbose:
            print(f"  Fold {i}:  LR AUC={lr_rows[-1]['auc']:.4f}  "
                  f"|  RF AUC={rf_rows[-1]['auc']:.4f}")

    lr_summary = _summarise(lr_rows)
    rf_summary = _summarise(rf_rows)

    lr_cv = lr_summary["auc"]["std"] / max(lr_summary["auc"]["mean"], 1e-9)
    rf_cv = rf_summary["auc"]["std"] / max(rf_summary["auc"]["mean"], 1e-9)
    lr_pass = lr_cv < CV_STD_THRESHOLD
    rf_pass = rf_cv < CV_STD_THRESHOLD

    result = {
        "dataset":       "lending_club",
        "k":             k,
        "cv_threshold":  CV_STD_THRESHOLD,
        "logistic_regression": {
            "folds":    lr_rows,
            "summary":  lr_summary,
            "auc_cv":   round(lr_cv, 4),
            "passed":   lr_pass,
        },
        "random_forest": {
            "folds":    rf_rows,
            "summary":  rf_summary,
            "auc_cv":   round(rf_cv, 4),
            "passed":   rf_pass,
        },
        "overall_passed": lr_pass and rf_pass,
    }

    os.makedirs(REPORT_DIR, exist_ok=True)
    out_path = os.path.join(REPORT_DIR, "cross_validation.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    if verbose:
        print()
        for name, s, cv, ok in [
            ("Logistic Regression", lr_summary, lr_cv, lr_pass),
            ("Random Forest",       rf_summary, rf_cv, rf_pass),
        ]:
            print(f"  {name}")
            print(f"    AUC       : {s['auc']['mean']:.4f} ± {s['auc']['std']:.4f}")
            print(f"    Precision : {s['precision']['mean']:.4f} ± {s['precision']['std']:.4f}")
            print(f"    Recall    : {s['recall']['mean']:.4f} ± {s['recall']['std']:.4f}")
            print(f"    F1        : {s['f1']['mean']:.4f} ± {s['f1']['std']:.4f}")
            print(f"    CV(AUC)   : {cv:.4f}   {'PASS' if ok else 'FAIL'}")
            print()

        print(f"  Report → {out_path}")
        print(f"  Overall: {'PASS' if result['overall_passed'] else 'FAIL'}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    p = argparse.ArgumentParser(description="K-fold CV on Lending Club")
    p.add_argument("--k", type=int, default=5, help="number of folds (default 5)")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    res = run(k=args.k, verbose=not args.quiet)
    sys.exit(0 if res["overall_passed"] else 1)


if __name__ == "__main__":
    _cli()
