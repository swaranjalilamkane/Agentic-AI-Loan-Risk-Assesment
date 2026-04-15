"""
Synthetic data augmentation for underrepresented groups – Task 5.

Uses SMOTE-style interpolation to generate synthetic borrower records for
protected-attribute groups that are too small for reliable fairness calibration.

Rules
-----
- Applied to TRAINING data only. The test set is never touched.
- Only generates samples for groups below `min_group_size`.
- Numerical features: interpolate between two random same-group samples.
- Categorical features: randomly pick value from one of the two seed samples.
- Engineered features (credit_amount_per_duration, credit_per_person) are
  recomputed from the interpolated base features so they stay consistent.
- Integer features are rounded; values clipped to the dataset min/max.

Usage
-----
    from src.data_integration.augment_data import augment_training_data

    df_aug = augment_training_data(df_train, group_col="personal_status_sex",
                                   min_group_size=150, random_state=42)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import logger

# Features recomputed from base columns after interpolation
DERIVED_FEATURES: dict[str, tuple[str, str]] = {
    "credit_amount_per_duration": ("amount", "duration"),
    "credit_per_person":          ("amount", "people_liable"),
}

# Numerical features to interpolate (exclude target, flags, derived)
INTERPOLATE_NUM = [
    "duration", "amount", "installment_rate",
    "present_residence", "age", "number_credits", "people_liable",
]

# Integer features — round after interpolation
INTEGER_FEATURES = [
    "duration", "amount", "installment_rate",
    "present_residence", "age", "number_credits", "people_liable",
]


def _interpolate_sample(
    row_a: pd.Series,
    row_b: pd.Series,
    num_cols: list[str],
    cat_cols: list[str],
    rng: np.random.Generator,
) -> dict:
    """
    Generate one synthetic sample by interpolating between row_a and row_b.

    Numerical: new = row_a + λ * (row_b - row_a)  where λ ~ Uniform(0, 1)
    Categorical: randomly pick from row_a or row_b
    """
    alpha = rng.random()
    new: dict = {}

    for col in num_cols:
        new[col] = float(row_a[col]) + alpha * (float(row_b[col]) - float(row_a[col]))
        if col in INTEGER_FEATURES:
            new[col] = int(round(new[col]))

    for col in cat_cols:
        new[col] = row_a[col] if rng.random() < 0.5 else row_b[col]

    return new


def augment_training_data(
    df_train: pd.DataFrame,
    group_col: str = "personal_status_sex",
    min_group_size: int = 150,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Augment the training dataframe so every group in `group_col` has at
    least `min_group_size` samples.

    Parameters
    ----------
    df_train       : training subset (after train/test split)
    group_col      : protected attribute column to balance
    min_group_size : target minimum count per group
    random_state   : reproducibility seed

    Returns
    -------
    Augmented dataframe (original rows + synthetic rows), shuffled.
    """
    rng      = np.random.default_rng(random_state)
    df_train = df_train.copy().reset_index(drop=True)

    # Determine which columns to interpolate vs keep categorical
    num_cols = [c for c in INTERPOLATE_NUM if c in df_train.columns]
    cat_cols = [
        c for c in df_train.columns
        if c not in num_cols
        and c not in DERIVED_FEATURES
        and c not in ("credit_risk", "_range_flag", "age_group")
        and c != group_col
    ]

    # Clip bounds per numerical column (stay within training distribution)
    col_bounds = {
        col: (df_train[col].min(), df_train[col].max())
        for col in num_cols
    }

    synthetic_rows: list[dict] = []
    group_counts = df_train[group_col].value_counts()

    for group, count in group_counts.items():
        if count >= min_group_size:
            continue                             # already large enough

        n_needed = min_group_size - count
        group_df = df_train[df_train[group_col] == group].reset_index(drop=True)

        if len(group_df) < 2:
            logger.warning(f"[augment] Group '{group}' has <2 samples — skipping.")
            continue

        logger.info(
            f"[augment] '{group}': {count} → {min_group_size} "
            f"(generating {n_needed} synthetic samples)"
        )

        for _ in range(n_needed):
            idx_a, idx_b = rng.choice(len(group_df), size=2, replace=False)
            row_a = group_df.iloc[idx_a]
            row_b = group_df.iloc[idx_b]

            new = _interpolate_sample(row_a, row_b, num_cols, cat_cols, rng)

            # Fix the group column
            new[group_col] = group

            # Clip numerical values to training range
            for col in num_cols:
                lo, hi = col_bounds[col]
                new[col] = int(max(lo, min(hi, new[col]))) if col in INTEGER_FEATURES \
                           else float(max(lo, min(hi, new[col])))

            # Recompute derived features
            for feat, (num, denom) in DERIVED_FEATURES.items():
                if num in new and denom in new and new[denom] != 0:
                    new[feat] = new[num] / new[denom]
                else:
                    new[feat] = 0.0

            # Keep credit_risk from a randomly chosen seed sample
            new["credit_risk"] = int(row_a["credit_risk"] if rng.random() < 0.5
                                     else row_b["credit_risk"])

            # Keep _range_flag as False (synthetic data is always clean)
            if "_range_flag" in df_train.columns:
                new["_range_flag"] = False

            # Recompute age_group from interpolated age
            if "age_group" in df_train.columns and "age" in new:
                age = new["age"]
                for lo, hi in [(18,30),(30,40),(40,50),(50,60),(60,100)]:
                    if lo <= age <= hi:
                        new["age_group"] = f"({lo}, {hi}]"
                        break

            synthetic_rows.append(new)

    if not synthetic_rows:
        logger.info("[augment] All groups already meet min_group_size — no augmentation needed.")
        return df_train

    df_synthetic = pd.DataFrame(synthetic_rows, columns=df_train.columns)
    df_augmented = pd.concat([df_train, df_synthetic], ignore_index=True)
    df_augmented = df_augmented.sample(frac=1, random_state=random_state).reset_index(drop=True)

    logger.info(
        f"[augment] Training set: {len(df_train)} → {len(df_augmented)} rows "
        f"(+{len(synthetic_rows)} synthetic)"
    )
    return df_augmented
