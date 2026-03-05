import os
import pandas as pd
from src.utils.logger import logger

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def clean_lendingclub(df: pd.DataFrame) -> pd.DataFrame:
    # basic cleaning (customize based on your columns)
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.drop_duplicates()

    # example: remove rows with missing target if you have one
    # if "loan_status" in df.columns:
    #     df = df.dropna(subset=["loan_status"])

    return df


def clean_german_credit(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.drop_duplicates()
    return df


def run_dataset_pipeline():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    lending_path = os.path.join(RAW_DIR, "lendingclub.csv")
    german_path = os.path.join(RAW_DIR, "german_credit.csv")

    logger.info("Loading datasets...")
    lc = load_csv(lending_path)
    gc = load_csv(german_path)

    logger.info("Cleaning datasets...")
    lc_clean = clean_lendingclub(lc)
    gc_clean = clean_german_credit(gc)

    lc_out = os.path.join(PROCESSED_DIR, "lendingclub_clean.csv")
    gc_out = os.path.join(PROCESSED_DIR, "german_credit_clean.csv")

    lc_clean.to_csv(lc_out, index=False)
    gc_clean.to_csv(gc_out, index=False)

    logger.info(f"Saved: {lc_out} ({lc_clean.shape})")
    logger.info(f"Saved: {gc_out} ({gc_clean.shape})")


if __name__ == "__main__":
    run_dataset_pipeline()
