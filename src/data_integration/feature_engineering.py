import pandas as pd
import os

PROCESSED_DIR = "data/processed"
OUTPUT_DIR = "outputs/features"

def build_features():

    german = pd.read_csv(os.path.join(PROCESSED_DIR, "german_credit_clean.csv"))

    # Example engineered features
    german["age_group"] = pd.cut(german["age"], bins=[18,30,40,50,60,100])

    german["credit_amount_per_duration"] = german["amount"] / german["duration"]

    german["credit_per_person"] = german["amount"] / german["people_liable"]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    german.to_csv(os.path.join(OUTPUT_DIR, "engineered_features.csv"), index=False)

    print("Feature engineering complete.")


if __name__ == "__main__":
    build_features()
