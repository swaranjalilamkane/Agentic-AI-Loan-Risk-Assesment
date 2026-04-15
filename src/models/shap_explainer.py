"""
SHAP Explainability Module – Task 5.

Loads the trained Logistic Regression and Random Forest models saved by
evaluate.py and generates:

  Global explanations (whole test set)
  ├── summary_bar   : mean |SHAP| per feature  → which features matter most overall
  └── beeswarm      : SHAP value distribution  → direction + magnitude per feature

  Local explanations (individual borrowers)
  └── waterfall     : single-borrower decision breakdown (why approved / rejected)

  Bias analysis (protected attributes)
  └── protected_impact : SHAP values for age and personal_status_sex plotted
                         against group membership to show what is driving bias

All plots saved under outputs/reports/shap/
Summary JSON saved to outputs/reports/shap_report.json

Usage
-----
    python -m src.models.shap_explainer          # standalone
    # or called from run_pipeline.py Step 6
"""

from __future__ import annotations

import json
import os
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")           # non-interactive – no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

from src.models.credit_model import (
    FEATURE_DIR,
    MODEL_DIR,
    load_model,
    prepare_data,
)
from src.models.fairness import binarize_age
from src.utils.logger import logger

SHAP_DIR   = "outputs/reports/shap"
REPORT_DIR = "outputs/reports"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_fig(filename: str) -> str:
    os.makedirs(SHAP_DIR, exist_ok=True)
    path = os.path.join(SHAP_DIR, filename)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    return path


def _top_features(shap_vals: np.ndarray, feature_names: list[str], n: int = 10) -> list[dict]:
    """Return top-n features ranked by mean absolute SHAP value."""
    mean_abs = np.abs(shap_vals).mean(axis=0)
    idx      = np.argsort(mean_abs)[::-1][:n]
    return [
        {"feature": feature_names[i], "mean_abs_shap": round(float(mean_abs[i]), 4)}
        for i in idx
    ]


# ---------------------------------------------------------------------------
# 1. Global explanations
# ---------------------------------------------------------------------------

def _global_summary(
    shap_vals: np.ndarray,
    X_test: pd.DataFrame,
    model_label: str,
    feature_names: list[str],
) -> tuple[str, str]:
    """
    Generate bar chart (feature importance) and beeswarm (value distribution).
    Returns paths to saved plots.
    """
    # -- Bar chart (mean |SHAP|) --
    shap.summary_plot(
        shap_vals, X_test,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        max_display=15,
    )
    plt.title(f"{model_label} — Global Feature Importance (mean |SHAP|)", fontsize=11)
    bar_path = _save_fig(f"{model_label.lower().replace(' ', '_')}_summary_bar.png")

    # -- Beeswarm (direction + spread) --
    shap.summary_plot(
        shap_vals, X_test,
        feature_names=feature_names,
        show=False,
        max_display=15,
    )
    plt.title(f"{model_label} — SHAP Value Distribution (beeswarm)", fontsize=11)
    bee_path = _save_fig(f"{model_label.lower().replace(' ', '_')}_summary_beeswarm.png")

    logger.info(f"[SHAP] Global plots saved for {model_label}.")
    return bar_path, bee_path


# ---------------------------------------------------------------------------
# 2. Local (per-borrower) explanations
# ---------------------------------------------------------------------------

def _local_waterfall(
    explainer,
    shap_vals: np.ndarray,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    model_label: str,
    feature_names: list[str],
    n_examples: int = 3,
) -> list[str]:
    """
    Generate waterfall plots for n_examples borrowers:
      - 1 true negative  (good credit, correctly approved)
      - 1 true positive  (default,     correctly rejected)
      - 1 false negative (default,     wrongly approved) if any exist
    Returns list of saved plot paths.
    """
    paths   = []
    X_arr   = X_test.values
    y_arr   = y_test.values if hasattr(y_test, "values") else np.array(y_test)

    # Select representative borrowers
    candidates = {
        "correctly_approved":  np.where((y_arr == 0) & (y_pred == 0))[0],
        "correctly_rejected":  np.where((y_arr == 1) & (y_pred == 1))[0],
        "wrongly_approved":    np.where((y_arr == 1) & (y_pred == 0))[0],
    }

    for label, indices in candidates.items():
        if len(indices) == 0:
            continue
        idx = indices[0]  # take first matching borrower

        sv = shap_vals[idx]
        # Ensure 1D — guard against any shape issues
        sv = np.array(sv).flatten()

        # expected_value: list/array → class 1; scalar → use directly
        ev = explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            ev = np.array(ev).flatten()
            base_val = float(ev[1]) if len(ev) > 1 else float(ev[0])
        else:
            base_val = float(ev)

        explanation = shap.Explanation(
            values       = sv,
            base_values  = base_val,
            data         = X_arr[idx],
            feature_names= feature_names,
        )

        plt.figure(figsize=(10, 5))
        shap.plots.waterfall(explanation, show=False, max_display=12)
        plt.title(
            f"{model_label} — Local Explanation\n"
            f"Borrower: {label.replace('_', ' ').title()}  "
            f"(true={'default' if y_arr[idx]==1 else 'good'}, "
            f"pred={'default' if y_pred[idx]==1 else 'good'})",
            fontsize=10,
        )
        fname = f"{model_label.lower().replace(' ', '_')}_local_{label}.png"
        paths.append(_save_fig(fname))

    logger.info(f"[SHAP] {len(paths)} local waterfall plots saved for {model_label}.")
    return paths


# ---------------------------------------------------------------------------
# 3. Bias / protected-attribute analysis
# ---------------------------------------------------------------------------

def _protected_impact(
    shap_vals: np.ndarray,
    X_test: pd.DataFrame,
    feature_names: list[str],
    raw_age: pd.Series,
    raw_sex: pd.Series,
    model_label: str,
) -> str:
    """
    Two-panel plot:
      Left  – SHAP(age) coloured by age group (≤25 vs >25)
      Right – mean SHAP(personal_status_sex) per sex/marital group (bar)

    Shows exactly how much age and sex are pushing decisions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"{model_label} — Protected Attribute SHAP Impact\n"
        "(positive SHAP = pushes toward default/reject; negative = pushes toward approval)",
        fontsize=10,
    )

    fn = list(feature_names)

    # --- Panel 1: age SHAP scatter ---
    if "age" in fn:
        age_idx       = fn.index("age")
        age_shap_vals = shap_vals[:, age_idx]
        age_bin       = binarize_age(raw_age).values
        colors        = ["#E74C3C" if g == "age<=25" else "#2980B9" for g in age_bin]

        axes[0].scatter(
            raw_age.values, age_shap_vals,
            c=colors, alpha=0.6, edgecolors="none", s=30,
        )
        axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
        axes[0].set_xlabel("Applicant Age", fontsize=10)
        axes[0].set_ylabel("SHAP value for age feature", fontsize=10)
        axes[0].set_title("Age SHAP values (red = ≤25, blue = >25)", fontsize=10)

        # annotate means
        for grp, col in [("age<=25", "#E74C3C"), ("age>25", "#2980B9")]:
            mask = age_bin == grp
            if mask.any():
                mean_sv = age_shap_vals[mask].mean()
                axes[0].axhline(mean_sv, color=col, linewidth=1.2,
                                linestyle=":", label=f"mean {grp}: {mean_sv:+.3f}")
        axes[0].legend(fontsize=8)
    else:
        axes[0].text(0.5, 0.5, "'age' not in features", ha="center", va="center")

    # --- Panel 2: sex group mean SHAP bar ---
    if "personal_status_sex" in fn:
        sex_idx       = fn.index("personal_status_sex")
        sex_shap_vals = shap_vals[:, sex_idx]
        sex_df = pd.DataFrame({"group": raw_sex.values, "shap": sex_shap_vals})
        grp_means = sex_df.groupby("group")["shap"].mean().sort_values()

        bar_colors = ["#E74C3C" if v > 0 else "#27AE60" for v in grp_means.values]
        axes[1].barh(grp_means.index, grp_means.values, color=bar_colors, alpha=0.8)
        axes[1].axvline(0, color="black", linewidth=0.8, linestyle="--")
        axes[1].set_xlabel("Mean SHAP value", fontsize=10)
        axes[1].set_title(
            "Mean SHAP(personal_status_sex) per group\n"
            "(red = pushes toward reject, green = pushes toward approval)",
            fontsize=9,
        )
        for i, (v, label) in enumerate(zip(grp_means.values, grp_means.index)):
            axes[1].text(v + 0.002 * np.sign(v), i, f"{v:+.3f}", va="center", fontsize=8)
    else:
        axes[1].text(0.5, 0.5, "'personal_status_sex' not in features", ha="center", va="center")

    plt.tight_layout()
    path = _save_fig(f"{model_label.lower().replace(' ', '_')}_protected_impact.png")
    logger.info(f"[SHAP] Protected attribute impact plot saved for {model_label}.")
    return path


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------

def run_shap_analysis() -> dict:
    """
    Load saved models, recreate the exact same train/test split, compute
    SHAP values, generate all plots, and save shap_report.json.

    Returns the shap_report dict.
    """
    print("=" * 60)
    print("Task 5: SHAP Explainability Analysis")
    print("=" * 60)

    # ── 1. Load models ───────────────────────────────────────────────────────
    print("\n[1/5] Loading trained models …")
    lr      = load_model("logistic_regression.pkl")
    rf      = load_model("random_forest.pkl")
    scaler  = load_model("scaler.pkl")
    print("      LR, RF, scaler loaded from outputs/models/")

    # ── 2. Recreate exact train/test split (same random_state=42) ────────────
    print("[2/5] Recreating train/test split …")
    feature_path = os.path.join(FEATURE_DIR, "engineered_features.csv")
    df = pd.read_csv(feature_path)

    # Preserve raw protected columns before encoding
    raw_sex = df["personal_status_sex"].copy() if "personal_status_sex" in df.columns else None
    raw_age = df["age"].copy() if "age" in df.columns else None

    X, y, _, feature_names = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    test_idx = X_test.index
    sex_test = raw_sex.loc[test_idx].reset_index(drop=True) if raw_sex is not None else None
    age_test = raw_age.loc[test_idx].reset_index(drop=True) if raw_age is not None else None

    X_test_reset  = X_test.reset_index(drop=True)
    X_train_reset = X_train.reset_index(drop=True)
    y_test_reset  = y_test.reset_index(drop=True)

    X_test_scaled  = scaler.transform(X_test_reset)
    X_train_scaled = scaler.transform(X_train_reset)
    print(f"      Test set: {len(X_test_reset)} samples | Features: {len(feature_names)}")

    # ── 3. Compute SHAP values ───────────────────────────────────────────────
    print("[3/5] Computing SHAP values (this may take a moment) …")

    # Random Forest — TreeExplainer (fast, exact for trees)
    rf_explainer  = shap.TreeExplainer(rf)
    rf_shap_raw   = rf_explainer.shap_values(X_test_reset)
    # Handle all shap version return formats:
    #   list   [class0, class1]         → shape (n_samples, n_features) each
    #   3D arr (n_samples, n_features, n_classes) → slice [:, :, 1]
    #   2D arr (n_samples, n_features)  → use as-is
    if isinstance(rf_shap_raw, list):
        rf_shap_vals = rf_shap_raw[1]
    elif isinstance(rf_shap_raw, np.ndarray) and rf_shap_raw.ndim == 3:
        rf_shap_vals = rf_shap_raw[:, :, 1]
    else:
        rf_shap_vals = rf_shap_raw
    print("      RF SHAP done  (TreeExplainer)")

    # Logistic Regression — LinearExplainer (fast, exact for linear models)
    lr_explainer  = shap.LinearExplainer(lr, X_train_scaled)
    lr_shap_vals  = lr_explainer.shap_values(X_test_scaled)
    if isinstance(lr_shap_vals, list):
        lr_shap_vals = lr_shap_vals[1]
    print("      LR SHAP done  (LinearExplainer)")

    # ── 4. Generate plots ────────────────────────────────────────────────────
    print("[4/5] Generating plots …")
    os.makedirs(SHAP_DIR, exist_ok=True)

    # — Global: RF —
    rf_bar, rf_bee = _global_summary(
        rf_shap_vals, X_test_reset, "Random Forest", feature_names
    )
    # — Global: LR (use scaled data so values are in same space) —
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names)
    lr_bar, lr_bee = _global_summary(
        lr_shap_vals, X_test_scaled_df, "Logistic Regression", feature_names
    )

    # — Local: RF waterfalls —
    rf_pred  = rf.predict(X_test_reset.values)
    rf_local = _local_waterfall(
        rf_explainer, rf_shap_vals, X_test_reset,
        y_test_reset, rf_pred, "Random Forest", feature_names,
    )

    # — Local: LR waterfalls —
    lr_pred  = lr.predict(X_test_scaled)
    lr_local = _local_waterfall(
        lr_explainer, lr_shap_vals, X_test_scaled_df,
        y_test_reset, lr_pred, "Logistic Regression", feature_names,
    )

    # — Protected attribute impact —
    rf_prot = lr_prot = None
    if sex_test is not None and age_test is not None:
        rf_prot = _protected_impact(
            rf_shap_vals, X_test_reset, feature_names,
            age_test, sex_test, "Random Forest",
        )
        lr_prot = _protected_impact(
            lr_shap_vals, X_test_scaled_df, feature_names,
            age_test, sex_test, "Logistic Regression",
        )

    # ── 5. Build & save report ───────────────────────────────────────────────
    print("[5/5] Saving SHAP report …")

    rf_top = _top_features(rf_shap_vals, feature_names)
    lr_top = _top_features(lr_shap_vals, feature_names)

    # Identify which protected features appear in top 10
    protected = {"age", "personal_status_sex"}
    rf_protected_ranks = {
        f["feature"]: rank + 1
        for rank, f in enumerate(rf_top)
        if f["feature"] in protected
    }
    lr_protected_ranks = {
        f["feature"]: rank + 1
        for rank, f in enumerate(lr_top)
        if f["feature"] in protected
    }

    report = {
        "random_forest": {
            "top_features":       rf_top,
            "protected_ranks":    rf_protected_ranks,
            "global_bar_plot":    rf_bar,
            "global_beeswarm":    rf_bee,
            "local_plots":        rf_local,
            "protected_plot":     rf_prot,
        },
        "logistic_regression": {
            "top_features":       lr_top,
            "protected_ranks":    lr_protected_ranks,
            "global_bar_plot":    lr_bar,
            "global_beeswarm":    lr_bee,
            "local_plots":        lr_local,
            "protected_plot":     lr_prot,
        },
    }

    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.join(REPORT_DIR, "shap_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # ── Console summary ──────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("SHAP Summary")
    print(f"{'─'*60}")

    for model_label, top in [("Random Forest", rf_top), ("Logistic Regression", lr_top)]:
        print(f"\n  {model_label} — Top 5 features by mean |SHAP|:")
        for i, f in enumerate(top[:5], 1):
            flag = "  ⚠ protected" if f["feature"] in protected else ""
            print(f"    {i}. {f['feature']:<35} {f['mean_abs_shap']:.4f}{flag}")

    print(f"\n  Protected attribute ranks:")
    for model_label, ranks in [("RF", rf_protected_ranks), ("LR", lr_protected_ranks)]:
        for feat, rank in ranks.items():
            print(f"    [{model_label}] '{feat}' is #{rank} most impactful feature")

    print(f"\n SHAP analysis complete.")
    print(f"   Plots  → {SHAP_DIR}/")
    print(f"   Report → {report_path}")

    return report


if __name__ == "__main__":
    run_shap_analysis()
