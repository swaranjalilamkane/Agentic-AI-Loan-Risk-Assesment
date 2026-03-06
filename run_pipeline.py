from src.data_integration.datasets import run_pipeline as run_datasets
from src.data_integration.feature_engineering import build_features
from src.data_integration.build_borrower_profile import create_profiles

def main():
    print("Step 1/3: Cleaning datasets...")
    run_datasets()

    print("Step 2/3: Feature engineering...")
    build_features()

    print("Step 3/3: Building borrower profiles...")
    create_profiles()

    print("✅ Pipeline completed successfully!")

if __name__ == "__main__":
    main()
