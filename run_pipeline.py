from src.data_integration.datasets import run_pipeline as run_datasets
from src.data_integration.data_validation import run_validation
from src.data_integration.feature_engineering import build_features
from src.data_integration.build_borrower_profile import create_profiles
from src.models.evaluate import run_evaluation
from src.models.shap_explainer import run_shap_analysis
from src.models.explain_borrower import explain_all_borrowers


def main():
    print("Step 1/7: Cleaning datasets...")
    run_datasets()

    print("\nStep 2/7: Validating datasets...")
    run_validation()

    print("\nStep 3/7: Feature engineering...")
    build_features()

    print("\nStep 4/7: Building borrower profiles...")
    create_profiles()

    print("\nStep 5/7: Training models, finding fairness thresholds and evaluating...")
    run_evaluation()

    print("\nStep 6/7: Running SHAP explainability analysis...")
    run_shap_analysis()

    print("\nStep 7/7: Generating human-readable borrower explanations (RF)...")
    explain_all_borrowers(model="rf", save_json=True)

    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()
