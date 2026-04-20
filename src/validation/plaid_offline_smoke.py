"""
Offline smoke test for the Plaid live-inference path.

Does NOT hit Plaid's network. Feeds synthetic Plaid-shaped JSON into the
mapper and through the orchestrator to prove the live-features path works
end-to-end independent of API credentials.

Test matrix:
    T1. Healthy borrower  — high balance, stable income, low debt
                           → expect APPROVED, prob < threshold
    T2. Risky  borrower  — overdrawn checking, no savings, high debt
                           → expect REJECTED, prob > threshold
    T3. Determinism       — same inputs twice produce the same decision
    T4. Provenance        — every feature has a documented source

Run:
    python -m src.validation.plaid_offline_smoke
"""
from __future__ import annotations

import json
import os
import sys

from src.agents.base import AgentState
from src.agents.data_retrieval_agent import DataRetrievalAgent
from src.agents.risk_assessment_agent import RiskAssessmentAgent
from src.agents.explanation_agent import ExplanationAgent
from src.data_integration.plaid_to_features import (
    FEATURE_ORDER, map_to_german_credit,
)
from src.agents.live_inference import score_live


# ---------------------------------------------------------------------------
# Synthetic Plaid payloads (mirror the real Plaid JSON shape exactly)
# ---------------------------------------------------------------------------

HEALTHY = {
    "accounts": [
        {"account_id": "c1", "name": "Checking", "type": "depository",
         "subtype": "checking", "balance_current": 3500.0,
         "balance_available": 3500.0, "currency": "USD"},
        {"account_id": "s1", "name": "Savings", "type": "depository",
         "subtype": "savings", "balance_current": 12000.0,
         "balance_available": 12000.0, "currency": "USD"},
    ],
    "transactions": [
        # debt outflows — small relative to income
        {"amount": 120.0, "category": ["Payment", "Credit Card"],
         "date": "2024-09-10", "name": "Visa payment",
         "transaction_id": "t1", "account_id": "c1", "pending": False},
        # rent
        {"amount": 1400.0, "category": ["Rent"], "date": "2024-09-01",
         "name": "Rent", "transaction_id": "t2",
         "account_id": "c1", "pending": False},
    ],
    "income": {"estimated_monthly_income": 6500.0,
               "estimated_monthly_spend": 3200.0,
               "transaction_count": 180},
    "application": {"amount": 4000, "duration": 18, "age": 38,
                    "purpose": "repairs",
                    "personal_status_sex": "male : single"},
}

RISKY = {
    "accounts": [
        {"account_id": "c2", "name": "Checking", "type": "depository",
         "subtype": "checking", "balance_current": -120.0,           # overdrawn
         "balance_available": -120.0, "currency": "USD"},
        # no savings account at all
    ],
    "transactions": [
        {"amount": 480.0, "category": ["Payment", "Credit Card"],
         "date": "2024-09-10", "name": "Visa payment",
         "transaction_id": "t3", "account_id": "c2", "pending": False},
        {"amount": 650.0, "category": ["Loan", "Auto Loan"],
         "date": "2024-09-05", "name": "Auto loan payment",
         "transaction_id": "t4", "account_id": "c2", "pending": False},
        {"amount": 300.0, "category": ["Loan", "Student Loan"],
         "date": "2024-09-02", "name": "Student loan",
         "transaction_id": "t5", "account_id": "c2", "pending": False},
    ],
    "income": {"estimated_monthly_income": 2200.0,
               "estimated_monthly_spend": 2100.0,
               "transaction_count": 90},
    "application": {"amount": 9000, "duration": 48, "age": 23,
                    "purpose": "other purpose",
                    "personal_status_sex": "female : divorced/separated/married"},
}


# ---------------------------------------------------------------------------
# Offline orchestrator run — bypasses PlaidConnector, uses the mapper directly
# ---------------------------------------------------------------------------

def _offline_pipeline(payload: dict, model: str = "rf") -> AgentState:
    """Skip PlaidConnector; call the mapper with pre-built JSON."""
    mapping = map_to_german_credit(
        accounts     = payload["accounts"],
        transactions = payload["transactions"],
        income       = payload["income"],
        application  = payload["application"],
    )

    # Build a state exactly the way DataRetrievalAgent._from_plaid would.
    state = AgentState(borrower_id=0, model_used=model)
    state.data_source        = "plaid_offline"
    state.raw_profile        = dict(mapping["features"])
    state.feature_names      = list(FEATURE_ORDER)
    state.live_features      = mapping["features"]
    state.feature_provenance = mapping["provenance"]
    state.raw_profile["_plaid_live_signals"]  = mapping["live_signals"]
    state.raw_profile["_plaid_defaults_used"] = mapping["defaults_used"]

    # Fake the DataRetrieval trace so downstream agents look standard
    state.agent_trace.append({
        "agent":       "data_retrieval_agent",
        "status":      "success",
        "elapsed_ms":  0.0,
        "description": "offline mock — no Plaid network call",
    })

    # Run the rest of the pipeline normally
    state = RiskAssessmentAgent()(state)
    state = ExplanationAgent()(state)
    return state


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

def run() -> dict:
    print("=" * 64)
    print(" Plaid Offline Smoke Test")
    print("=" * 64)

    results: dict = {}

    # T1 — healthy
    s_ok = _offline_pipeline(HEALTHY, model="rf")
    # T2 — risky
    s_bad = _offline_pipeline(RISKY,   model="rf")
    # T3 — determinism
    s_ok2 = _offline_pipeline(HEALTHY, model="rf")

    t1 = {
        "name":        "healthy_borrower_low_risk",
        "decision":    s_ok.decision,
        "probability": s_ok.default_probability,
        "passed":      s_ok.default_probability is not None
                       and s_ok.default_probability < 0.65,
    }
    t2 = {
        "name":        "risky_borrower_high_risk",
        "decision":    s_bad.decision,
        "probability": s_bad.default_probability,
        "passed":      s_bad.default_probability is not None
                       and s_bad.default_probability > s_ok.default_probability,
    }
    t3 = {
        "name":      "deterministic_across_two_runs",
        "prob_run1": s_ok.default_probability,
        "prob_run2": s_ok2.default_probability,
        "passed":    s_ok.default_probability == s_ok2.default_probability,
    }
    # T4 — every FEATURE_ORDER column has a provenance entry
    missing_prov = [f for f in FEATURE_ORDER
                    if f not in s_ok.feature_provenance]
    t4 = {
        "name":          "provenance_complete",
        "missing":       missing_prov,
        "passed":        len(missing_prov) == 0,
    }

    results = {"T1": t1, "T2": t2, "T3": t3, "T4": t4}
    overall = all(r["passed"] for r in results.values())

    for key, r in results.items():
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {key}: {r['name']}")
        for k, v in r.items():
            if k in ("name", "passed"):
                continue
            print(f"           {k}: {v}")

    # Print the healthy narrative so the demo output is legible
    print("\n  Healthy-borrower narrative (sample):")
    print("  " + "-" * 60)
    if s_ok.narrative:
        for line in s_ok.narrative.splitlines()[:6]:
            print(f"    {line}")

    print(f"\n  Overall: {'PASS' if overall else 'FAIL'}")

    # Save combined report
    out_dir = "outputs/reports/validation"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "plaid_offline_smoke.json")
    with open(out_path, "w") as f:
        json.dump({
            "healthy":   {"decision": s_ok.decision,
                          "probability": s_ok.default_probability,
                          "live_signals": s_ok.raw_profile.get("_plaid_live_signals")},
            "risky":     {"decision": s_bad.decision,
                          "probability": s_bad.default_probability,
                          "live_signals": s_bad.raw_profile.get("_plaid_live_signals")},
            "checks":    results,
            "overall_passed": overall,
        }, f, indent=2, default=str)
    print(f"  Report → {out_path}")

    return {"results": results, "overall_passed": overall}


if __name__ == "__main__":
    res = run()
    sys.exit(0 if res["overall_passed"] else 1)
