import pandas as pd
import json
import os

FEATURE_DIR = "outputs/features"
PROFILE_DIR = "outputs/borrower_profiles"


def create_profiles():

    df = pd.read_csv(os.path.join(FEATURE_DIR, "engineered_features.csv"))

    os.makedirs(PROFILE_DIR, exist_ok=True)

    for i, row in df.head(50).iterrows():

        profile = row.to_dict()

        with open(os.path.join(PROFILE_DIR, f"borrower_{i}.json"), "w") as f:
            json.dump(profile, f, indent=4)

    print("Borrower profiles generated.")


if __name__ == "__main__":
    create_profiles()
