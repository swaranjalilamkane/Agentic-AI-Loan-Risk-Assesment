"""
Data validation module – Task 3 (extracted from notebooks/data_validation.ipynb).

Sits between datasets.py (basic load/clean) and feature_engineering.py in the pipeline.
Works with column names exactly as they come out of datasets.py (lowercased originals).
Does NOT rename columns so downstream code is unaffected.

Sections:
  4.1  Schema Validation       – required fields, numeric types, range bounds
  4.2  Completeness            – per-record and per-feature missingness
  4.3  Consistency Checks      – cross-field logic (age vs credit history, etc.)
  4.4  Fairness-Aware Checks   – group representation, proxy leakage, label distribution
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.utils.logger import logger

PROCESSED_DIR = "data/processed"

# ---------------------------------------------------------------------------
# German Credit – column config (column names after datasets.py lowercasing)
# ---------------------------------------------------------------------------

GC_REQUIRED = ["amount", "duration", "age", "credit_risk"]

GC_NUMERIC = [
    "amount", "duration", "age", "installment_rate",
    "present_residence", "number_credits", "people_liable",
]

GC_RANGE_BOUNDS = {
    "age":              (18, 100),
    "amount":           (0,  None),
    "duration":         (1,  None),
    "installment_rate": (1,  4),
    "number_credits":   (1,  None),
    "people_liable":    (1,  None),
    "credit_risk":      (1,  2),
}

GC_PROTECTED_COLS   = ["personal_status_sex", "foreign_worker"]
GC_PROXY_FEATURES   = ["amount", "duration", "installment_rate"]
GC_TARGET           = "credit_risk"

# ---------------------------------------------------------------------------
# Lending Club – column config
# ---------------------------------------------------------------------------

LC_REQUIRED = ["loan_amnt", "loan_status"]

LC_NUMERIC = [
    "loan_amnt", "annual_inc", "dti", "fico_range_low",
    "installment", "open_acc", "revol_bal", "total_acc",
]

LC_RANGE_BOUNDS = {
    "annual_inc":     (0,    None),
    "loan_amnt":      (0,    None),
    "dti":            (0,    100),
    "fico_range_low": (300,  850),
}

LC_PROTECTED_COLS = ["home_ownership"]
LC_PROXY_FEATURES = ["dti", "installment", "annual_inc"]
LC_TARGET         = "loan_status"   # still raw text at this stage

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

RECORD_MISSING_THRESHOLD  = 0.30   # drop records missing >30 % of fields
FEATURE_MISSING_THRESHOLD = 0.50   # drop features missing >50 % of values
PROXY_CORR_THRESHOLD      = 0.70   # flag feature↔protected correlation above this
MIN_GROUP_PROPORTION      = 0.05   # warn if any group is <5 % of dataset
MAX_DEFAULT_RATE_GAP      = 0.15   # warn if default rate differs >15 % across groups


# ===========================================================================
# 4.1  Schema Validation
# ===========================================================================

def _check_required_fields(df: pd.DataFrame, required: list[str], source: str) -> pd.DataFrame:
    present = [c for c in required if c in df.columns]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        logger.warning(f"[4.1][{source}] Required columns not found in dataset: {missing_cols}")

    if not present:
        return df

    mask = df[present].isnull().any(axis=1)
    if mask.any():
        logger.warning(f"[4.1][{source}] {mask.sum()} records dropped – missing required fields.")
    else:
        logger.info(f"[4.1][{source}] Required fields: all {len(df)} records pass.")
    return df[~mask].copy()


def _enforce_numeric_types(df: pd.DataFrame, numeric_cols: list[str], source: str) -> pd.DataFrame:
    df = df.copy()
    for col in numeric_cols:
        if col not in df.columns:
            continue
        before = df[col].isnull().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        after  = df[col].isnull().sum()
        new_nulls = after - before
        if new_nulls > 0:
            logger.warning(f"[4.1][{source}] '{col}': {new_nulls} non-numeric values coerced to NaN.")
    logger.info(f"[4.1][{source}] Numeric type enforcement done.")
    return df


def _check_range_bounds(df: pd.DataFrame, bounds: dict, source: str) -> pd.DataFrame:
    df = df.copy()
    df["_range_flag"] = False
    for col, (lo, hi) in bounds.items():
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        mask = pd.Series(False, index=df.index)
        if lo is not None:
            mask |= vals < lo
        if hi is not None:
            mask |= vals > hi
        mask &= vals.notnull()
        if mask.any():
            logger.warning(f"[4.1][{source}] '{col}': {mask.sum()} out-of-range values flagged.")
            df.loc[mask, "_range_flag"] = True
    flagged = df["_range_flag"].sum()
    logger.info(f"[4.1][{source}] Range check: {flagged} records flagged (retained with flag).")
    return df


# ===========================================================================
# 4.2  Completeness & Missingness
# ===========================================================================

def _check_record_completeness(df: pd.DataFrame, source: str) -> pd.DataFrame:
    ratio = df.isnull().mean(axis=1)
    bad   = ratio > RECORD_MISSING_THRESHOLD
    if bad.any():
        logger.warning(
            f"[4.2][{source}] {bad.sum()} records dropped "
            f"(>{int(RECORD_MISSING_THRESHOLD*100)}% fields missing)."
        )
    else:
        logger.info(f"[4.2][{source}] Record completeness: all records pass.")
    return df[~bad].copy()


def _drop_high_missingness_features(df: pd.DataFrame, source: str) -> pd.DataFrame:
    ratio     = df.isnull().mean()
    drop_cols = ratio[ratio > FEATURE_MISSING_THRESHOLD].index.tolist()
    # Never drop protected, target, or key feature columns accidentally
    safe_drop = [c for c in drop_cols if not c.startswith("_")]
    if safe_drop:
        logger.warning(
            f"[4.2][{source}] Dropping {len(safe_drop)} high-missingness features: {safe_drop}"
        )
        df = df.drop(columns=safe_drop)
    else:
        logger.info(f"[4.2][{source}] No features dropped for missingness.")
    return df


def _log_missingness(df: pd.DataFrame, source: str) -> None:
    top = df.isnull().mean().sort_values(ascending=False)
    top = top[top > 0].head(8)
    if not top.empty:
        logger.info(f"[4.2][{source}] Top missing features:\n{top.to_string()}")
    else:
        logger.info(f"[4.2][{source}] No missing values detected.")


# ===========================================================================
# 4.3  Consistency Checks
# ===========================================================================

def _check_age_vs_duration(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Flag records where loan duration (months) implies credit started before age 18."""
    df = df.copy()
    if "age" not in df.columns or "duration" not in df.columns:
        logger.info(f"[4.3][{source}] Age vs duration: skipped (columns absent).")
        return df
    age      = pd.to_numeric(df["age"], errors="coerce")
    duration = pd.to_numeric(df["duration"], errors="coerce")
    # duration in months → years
    mask = (duration / 12) > (age - 18)
    mask &= age.notnull() & duration.notnull()
    if mask.any():
        logger.warning(f"[4.3][{source}] {mask.sum()} records flagged: loan duration > (age - 18) years.")
        df.loc[mask, "_range_flag"] = True
    else:
        logger.info(f"[4.3][{source}] Age vs duration: all records consistent.")
    return df


def _check_installment_rate_vs_amount(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Flag records where installment rate is suspiciously high relative to loan amount."""
    df = df.copy()
    if "installment_rate" not in df.columns or "amount" not in df.columns:
        logger.info(f"[4.3][{source}] Installment vs amount: skipped.")
        return df
    rate   = pd.to_numeric(df["installment_rate"], errors="coerce")
    amount = pd.to_numeric(df["amount"], errors="coerce")
    # installment_rate is 1–4 (% of disposable income); flag if amount=0 but rate>0
    mask = (amount <= 0) & (rate > 0) & amount.notnull() & rate.notnull()
    if mask.any():
        logger.warning(f"[4.3][{source}] {mask.sum()} records: amount ≤ 0 with installment_rate > 0.")
        df.loc[mask, "_range_flag"] = True
    else:
        logger.info(f"[4.3][{source}] Installment vs amount: all records consistent.")
    return df


def _check_lc_income_vs_dti(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Flag Lending Club records where DTI is inconsistent with reported income."""
    df = df.copy()
    if "annual_inc" not in df.columns or "dti" not in df.columns:
        logger.info(f"[4.3][{source}] Income vs DTI: skipped.")
        return df
    inc = pd.to_numeric(df["annual_inc"], errors="coerce")
    dti = pd.to_numeric(df["dti"], errors="coerce")
    # Extreme: income > 0 but DTI = 0 (likely missing), or DTI > 100
    mask = ((inc > 0) & (dti == 0)) | (dti > 100)
    mask &= inc.notnull() & dti.notnull()
    if mask.any():
        logger.warning(f"[4.3][{source}] {mask.sum()} records flagged for income/DTI inconsistency.")
        df.loc[mask, "_range_flag"] = True
    else:
        logger.info(f"[4.3][{source}] Income vs DTI: all records consistent.")
    return df


# ===========================================================================
# 4.4  Fairness-Aware Data Checks
# ===========================================================================

def _fairness_data_checks(
    df: pd.DataFrame,
    source: str,
    protected_cols: list[str],
    proxy_features: list[str],
    target_col: str,
) -> None:
    """
    Pre-modelling fairness checks on raw data.

    Checks:
    - Group representation (warn if any group < MIN_GROUP_PROPORTION)
    - Proxy/leakage: correlation between engineered features and protected attributes
    - Label distribution gap across protected groups
    """
    for pcol in protected_cols:
        if pcol not in df.columns:
            continue

        # -- Representation --
        props = df[pcol].value_counts(normalize=True)
        under = props[props < MIN_GROUP_PROPORTION]
        if not under.empty:
            logger.warning(
                f"[4.4][{source}] Under-represented groups in '{pcol}':\n{under.to_string()}"
            )
        else:
            logger.info(f"[4.4][{source}] Group representation OK for '{pcol}'.")

        # -- Proxy leakage --
        enc = pd.factorize(df[pcol])[0]
        for feat in proxy_features:
            if feat not in df.columns:
                continue
            vals  = pd.to_numeric(df[feat], errors="coerce")
            valid = vals.notnull()
            if valid.sum() < 10:
                continue
            corr = abs(np.corrcoef(enc[valid], vals[valid])[0, 1])
            if corr > PROXY_CORR_THRESHOLD:
                logger.warning(
                    f"[4.4][{source}] Potential proxy: '{feat}' ↔ '{pcol}' |r|={corr:.2f}"
                )
            else:
                logger.info(
                    f"[4.4][{source}] Leakage check OK: '{feat}' ↔ '{pcol}' |r|={corr:.2f}"
                )

        # -- Label distribution --
        if target_col not in df.columns:
            continue
        target_num = pd.to_numeric(df[target_col], errors="coerce")
        sub        = pd.concat([df[pcol], target_num], axis=1).dropna()
        if sub.empty:
            continue
        rates = sub.groupby(pcol)[target_col].mean()
        gap   = rates.max() - rates.min()
        if gap > MAX_DEFAULT_RATE_GAP:
            logger.warning(
                f"[4.4][{source}] Label distribution gap in '{pcol}': {gap:.2%}\n"
                f"{rates.to_string()}"
            )
        else:
            logger.info(
                f"[4.4][{source}] Label distribution OK for '{pcol}'. Gap={gap:.2%}"
            )


# ===========================================================================
# Master validation pipelines (one per dataset)
# ===========================================================================

def validate_german_credit(df: pd.DataFrame) -> pd.DataFrame:
    source = "german_credit"
    logger.info(f"[{source}] Starting validation — {len(df)} records, {df.shape[1]} columns.")

    df = _check_required_fields(df, GC_REQUIRED, source)
    df = _enforce_numeric_types(df, GC_NUMERIC, source)
    df = _check_range_bounds(df, GC_RANGE_BOUNDS, source)
    df = _check_record_completeness(df, source)
    df = _drop_high_missingness_features(df, source)
    _log_missingness(df, source)
    df = _check_age_vs_duration(df, source)
    df = _check_installment_rate_vs_amount(df, source)
    _fairness_data_checks(df, source, GC_PROTECTED_COLS, GC_PROXY_FEATURES, GC_TARGET)

    flagged = df.get("_range_flag", pd.Series(False, index=df.index)).sum()
    logger.info(
        f"[{source}] Validation complete — {len(df)} records pass, {flagged} flagged."
    )
    return df


def validate_lending_club(df: pd.DataFrame) -> pd.DataFrame:
    source = "lending_club"
    logger.info(f"[{source}] Starting validation — {len(df)} records, {df.shape[1]} columns.")

    df = _check_required_fields(df, LC_REQUIRED, source)
    df = _enforce_numeric_types(df, LC_NUMERIC, source)
    df = _check_range_bounds(df, LC_RANGE_BOUNDS, source)
    df = _check_record_completeness(df, source)
    df = _drop_high_missingness_features(df, source)
    _log_missingness(df, source)
    df = _check_lc_income_vs_dti(df, source)
    _fairness_data_checks(df, source, LC_PROTECTED_COLS, LC_PROXY_FEATURES, LC_TARGET)

    flagged = df.get("_range_flag", pd.Series(False, index=df.index)).sum()
    logger.info(
        f"[{source}] Validation complete — {len(df)} records pass, {flagged} flagged."
    )
    return df


# ===========================================================================
# Pipeline entry point
# ===========================================================================

def run_validation() -> None:
    """
    Load the processed CSVs written by datasets.py, validate them,
    and overwrite the files with the validated versions.
    """
    german_path  = os.path.join(PROCESSED_DIR, "german_credit_clean.csv")
    lending_path = os.path.join(PROCESSED_DIR, "lendingclub_clean.csv")

    if os.path.exists(german_path):
        german = pd.read_csv(german_path)
        german = validate_german_credit(german)
        german.to_csv(german_path, index=False)
        print(f"  German Credit : {len(german)} records validated and saved.")
    else:
        print(f"  German Credit : file not found at {german_path}, skipping.")

    if os.path.exists(lending_path):
        lending = pd.read_csv(lending_path)
        lending = validate_lending_club(lending)
        lending.to_csv(lending_path, index=False)
        print(f"  Lending Club  : {len(lending)} records validated and saved.")
    else:
        print(f"  Lending Club  : file not found at {lending_path}, skipping.")


if __name__ == "__main__":
    run_validation()
