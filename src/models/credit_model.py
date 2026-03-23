"""
Credit risk ML models – Task 4.

Trains two classifiers on the German Credit dataset:
  1. Logistic Regression  (class_weight='balanced')
  2. Random Forest        (class_weight='balanced')

Both use balanced class weights to handle the class imbalance in the
German Credit dataset (~30 % bad / 70 % good credit).

Target encoding:
  credit_risk == 1  →  1  (bad credit / default)
  credit_risk == 2  →  0  (good credit / no default)

Usage
-----
    from src.models.credit_model import prepare_data, train_models, evaluate_model
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

FEATURE_DIR = "outputs/features"
MODEL_DIR = "outputs/models"


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict, list[str]]:
    """
    Encode and split the engineered features dataframe.

    Returns
    -------
    X : pd.DataFrame     – model input features (all numeric)
    y : pd.Series        – binary target (1 = default)
    encoders : dict      – fitted LabelEncoders keyed by column name
    feature_names : list – ordered feature names matching X.columns
    """
    df = df.copy()

    # Drop the interval age_group column (use raw age instead)
    if "age_group" in df.columns:
        df = df.drop(columns=["age_group"])

    # Binarise target: 1 (bad) → 1 (default), 2 (good) → 0 (no default)
    df["target"] = (df["credit_risk"] == 1).astype(int)
    df = df.drop(columns=["credit_risk"])

    # Encode all remaining object columns
    encoders: dict[str, LabelEncoder] = {}
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y, encoders, list(X.columns)


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[LogisticRegression, RandomForestClassifier, StandardScaler]:
    """
    Fit Logistic Regression and Random Forest classifiers.

    class_weight='balanced' adjusts weights inversely proportional to
    class frequencies to counteract the ~30/70 imbalance.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )
    lr.fit(X_scaled, y_train)

    rf = RandomForestClassifier(
        class_weight="balanced",
        n_estimators=100,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)   # RF doesn't require scaling

    return lr, rf, scaler


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    model_name: str,
    scaler: StandardScaler | None = None,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Compute AUC-ROC, precision, and recall for a fitted model.

    Parameters
    ----------
    scaler : optional – if provided, X_test is scaled before prediction
             (required for Logistic Regression).

    Returns
    -------
    metrics : dict
    y_pred  : np.ndarray of class predictions
    y_prob  : np.ndarray of positive-class probabilities
    """
    X_eval = scaler.transform(X_test) if scaler is not None else X_test

    y_pred = model.predict(X_eval)
    y_prob = model.predict_proba(X_eval)[:, 1]

    metrics = {
        "model": model_name,
        "auc_roc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "classification_report": classification_report(y_test, y_pred),
    }
    return metrics, y_pred, y_prob


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_model(obj, filename: str) -> str:
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return path


def load_model(filename: str):
    path = os.path.join(MODEL_DIR, filename)
    with open(path, "rb") as f:
        return pickle.load(f)
