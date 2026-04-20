"""
Validation package — four independent validation suites.

    cross_validation.py   – K-fold on Lending Club
    fairness_validation.py – Demographic Parity + Equalized Odds on German Credit
    explainability_validation.py – SHAP rankings vs. domain expectations
    pipeline_validation.py – End-to-end orchestrator smoke tests

Each module exposes a `run()` function returning a dict result and also
writes a JSON report under outputs/reports/validation/.
"""
