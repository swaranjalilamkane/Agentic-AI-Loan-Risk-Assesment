"""
Plaid Live-Inference Demo
=========================

End-to-end example of the agentic pipeline running against LIVE Plaid Sandbox
data instead of the German-Credit CSV test set.

Pipeline:
    DataRetrievalAgent(source="plaid")  ──► pulls accounts/txns/income
                                            from Plaid sandbox
    RiskAssessmentAgent                 ──► scores the live feature vector
    ExplanationAgent                    ──► generates SHAP narrative

Prerequisites:
    1. Export Plaid sandbox creds in your shell or .env
           export PLAID_CLIENT_ID=...
           export PLAID_SECRET=...
           export PLAID_ENV=sandbox
    2. Trained models exist at outputs/models/

Run:
    python plaid_live_demo.py                  # use default application form
    python plaid_live_demo.py --amount 8000 --duration 36 --age 42
    python plaid_live_demo.py --model lr
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from src.agents.base import AgentState
from src.agents.data_retrieval_agent import DataRetrievalAgent
from src.agents.risk_assessment_agent import RiskAssessmentAgent
from src.agents.explanation_agent import ExplanationAgent
from src.agents.orchestrator import Orchestrator


def _parse_args():
    p = argparse.ArgumentParser(description="Live Plaid inference demo")
    p.add_argument("--amount",              type=int, default=5000)
    p.add_argument("--duration",            type=int, default=24)
    p.add_argument("--age",                 type=int, default=35)
    p.add_argument("--purpose",             type=str, default="other purpose")
    p.add_argument("--personal-status-sex", type=str, default="male : single",
                   dest="personal_status_sex")
    p.add_argument("--model",               type=str, default="rf",
                   choices=["rf", "lr"])
    p.add_argument("--institution",         type=str, default="ins_109508",
                   help="Plaid sandbox institution id")
    p.add_argument("--save",                type=str, default=None,
                   help="Optional JSON output path")
    return p.parse_args()


def main():
    args = _parse_args()

    # Guard — cred check
    if not (os.getenv("PLAID_CLIENT_ID") and os.getenv("PLAID_SECRET")):
        print("✗ PLAID_CLIENT_ID / PLAID_SECRET not set in environment.")
        print("  Export them (or add to .env) and re-run.")
        sys.exit(2)

    application = {
        "amount":              args.amount,
        "duration":            args.duration,
        "age":                 args.age,
        "purpose":             args.purpose,
        "personal_status_sex": args.personal_status_sex,
    }

    print("=" * 64)
    print(" Plaid Live-Inference Demo")
    print("=" * 64)
    print(f"  Institution : {args.institution}  (Plaid sandbox)")
    print(f"  Model       : {args.model.upper()}")
    print(f"  Application : {application}")
    print()

    # Custom orchestrator wired with a Plaid-enabled DataRetrievalAgent
    orch = Orchestrator(agents=[
        DataRetrievalAgent(
            source         = "plaid",
            application    = application,
            institution_id = args.institution,
        ),
        RiskAssessmentAgent(),
        ExplanationAgent(),
    ])

    # borrower_id is required by AgentState contract but unused for live Plaid
    state = orch.run(borrower_id=0, model=args.model, verbose=True)
    Orchestrator.print_report(state)

    # Live-signal breakdown
    live = state.raw_profile.get("_plaid_live_signals", {})
    defs = state.raw_profile.get("_plaid_defaults_used", [])
    print(" Plaid Live Signals:")
    print(" " + "-" * 60)
    for k, v in live.items():
        print(f"   {k:<22} {v}")
    if defs:
        print(f"\n  Application defaults used for: {', '.join(defs)}")
    print()

    # Provenance table
    if state.feature_provenance:
        print(" Feature Provenance:")
        print(" " + "-" * 60)
        src_groups: dict[str, list[str]] = {}
        for feat, src_ in state.feature_provenance.items():
            src_groups.setdefault(src_, []).append(feat)
        for src_name, feats in src_groups.items():
            print(f"   {src_name:<12} ({len(feats)}): {', '.join(feats)}")
        print()

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        # Drop non-serialisable numpy scalars by funneling through json default=str
        with open(args.save, "w") as f:
            json.dump(state.to_dict(), f, indent=2, default=str)
        print(f"  Saved → {args.save}")


if __name__ == "__main__":
    main()
