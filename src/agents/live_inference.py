"""
Live-inference helper — score an ad-hoc feature dict through the trained
LR / RF models.

The standard pipeline uses *pre-computed* probabilities on the 250-row test
set (indexed by `borrower_id`). For live Plaid-sourced borrowers there is no
precomputed probability, so this helper runs the model forward on the fly.

    score_live(features_dict, model="rf") -> (probability, prediction)

Feature-order, encoding, and scaling must exactly match training:
  1. Use the same feature column order as `prepare_data` produces.
  2. Re-use the saved `encoders.pkl` LabelEncoders for all categorical columns.
  3. Apply the saved `scaler.pkl` for LR only.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from src.models.credit_model import FEATURE_DIR, load_model, prepare_data


@lru_cache(maxsize=1)
def _artifacts():
    """Load models once per process."""
    lr      = load_model("logistic_regression.pkl")
    rf      = load_model("random_forest.pkl")
    scaler  = load_model("scaler.pkl")

    # Recover the exact feature order and encoders used at training time
    import os
    df = pd.read_csv(f"{FEATURE_DIR}/engineered_features.csv")
    _, _, encoders, feature_names = prepare_data(df)
    return {
        "lr":            lr,
        "rf":            rf,
        "scaler":        scaler,
        "encoders":      encoders,
        "feature_names": feature_names,
    }


def _encode_row(features: dict) -> np.ndarray:
    """Turn a feature dict into a 1×N numpy row matching training order."""
    art           = _artifacts()
    feature_names = art["feature_names"]
    encoders      = art["encoders"]

    row = []
    for col in feature_names:
        val = features.get(col)
        if val is None:
            raise ValueError(f"Missing feature: {col!r}")
        if col in encoders:
            le = encoders[col]
            s = str(val)
            # Unseen category → fallback to first class (documented)
            if s not in set(le.classes_):
                val_enc = 0
            else:
                val_enc = int(le.transform([s])[0])
            row.append(val_enc)
        else:
            row.append(float(val))
    return np.array([row], dtype=float)


def score_live(features: dict, model: str = "rf") -> tuple[float, int]:
    """
    Compute default-probability + class prediction for a single feature dict.

    Parameters
    ----------
    features : dict with all 22 engineered-feature columns
    model    : "rf" or "lr"

    Returns
    -------
    probability : float – predicted P(default)
    prediction  : int   – 1 (default / reject) or 0 (good / approve)
                          using the model's default 0.5 threshold (fairness
                          threshold is applied later by RiskAssessmentAgent)
    """
    if model not in ("rf", "lr"):
        raise ValueError(f"model must be 'rf' or 'lr', got {model!r}")

    art = _artifacts()
    X = _encode_row(features)

    if model == "lr":
        X = art["scaler"].transform(X)
        clf = art["lr"]
    else:
        clf = art["rf"]

    prob = float(clf.predict_proba(X)[0, 1])
    pred = int(prob >= 0.5)
    return prob, pred
