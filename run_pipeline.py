from src.data_integration.datasets import run_pipeline as run_datasets
from src.data_integration.feature_engineering import build_features
from src.data_integration.build_borrower_profile import create_profiles
from src.models.evaluate import run_evaluation


def main():
    print("Step 1/4: Cleaning datasets...")
    run_datasets()

    print("\nStep 2/4: Feature engineering...")
    build_features()

    print("\nStep 3/4: Building borrower profiles...")
    create_profiles()

    print("\nStep 4/4: Training credit risk models and evaluating fairness...")
    run_evaluation()

    print("\n✅ Pipeline completed successfully!")


if __name__ == "__main__":
    main()
