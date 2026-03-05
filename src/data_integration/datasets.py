import pandas as pd
import os

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

def load_datasets():
    german_path = os.path.join(RAW_DIR, "german_credit.csv")
    lending_path = os.path.join(RAW_DIR, "lendingclub.csv")

    german = pd.read_csv(german_path)
    lending = pd.read_csv(lending_path)

    return german, lending


def clean_dataset(df):
    df = df.copy()

    # remove duplicates
    df = df.drop_duplicates()

    # normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    return df


def save_processed(german, lending):
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    german.to_csv(os.path.join(PROCESSED_DIR, "german_credit_clean.csv"), index=False)
    lending.to_csv(os.path.join(PROCESSED_DIR, "lendingclub_clean.csv"), index=False)


def run_pipeline():
    german, lending = load_datasets()

    german = clean_dataset(german)
    lending = clean_dataset(lending)

    save_processed(german, lending)

    print("Datasets cleaned and saved in data/processed/")


if __name__ == "__main__":
    run_pipeline()
