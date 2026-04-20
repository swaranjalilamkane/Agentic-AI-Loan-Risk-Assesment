"""
Pipeline Validation — End-to-End Orchestrator Smoke Tests.

Goal
----
Confirm the multi-agent orchestrator:

    Data Retrieval Agent  →  Risk Assessment Agent  →  Explanation Agent

produces consistent, auditable outputs:

    T1. Every borrower ID in the sample runs without errors.
    T2. Every run executes all 3 agents in order (full audit trail).
    T3. Every agent emits a success status + non-null elapsed_ms.
    T4. Required output fields are populated (prob, decision, narrative, factors).
    T5. Decision is consistent with probability + threshold.
    T6. Determinism — running the same borrower twice yields identical outputs.
    T7. Choosing LR vs RF yields different-but-valid outputs for at least one
        borrower (sanity: models aren't silently returning the same thing).

Usage
-----
    python -m src.validation.pipeline_validation
    python -m src.validation.pipeline_validation --ids 1 10 42 100 250 500
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from dataclasses import asdict

warnings.filterwarnings("ignore")

from src.agents.orchestrator import Orchestrator

REPORT_DIR = "outputs/reports/validation"

DEFAULT_SAMPLE_IDS = [1, 25, 50, 100, 150, 200]  # test set has 250 rows
EXPECTED_AGENTS = [
    "data_retrieval_agent",
    "risk_assessment_agent",
    "explanation_agent",
]
REQUIRED_FIELDS = [
    "default_probability",
    "decision",
    "risk_level",
    "narrative",
    "risk_factors",
    "protective_factors",
    "shap_report",
]


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_no_errors(state) -> tuple[bool, str]:
    if state.errors:
        return False, f"errors present: {state.errors}"
    return True, "no errors"


def _check_agent_order(state) -> tuple[bool, str]:
    got = [t["agent"] for t in state.agent_trace]
    if got != EXPECTED_AGENTS:
        return False, f"agent order mismatch: expected {EXPECTED_AGENTS}, got {got}"
    return True, "correct agent order"


def _check_agent_success(state) -> tuple[bool, str]:
    for t in state.agent_trace:
        if t.get("status") != "success":
            return False, f"{t['agent']} status={t.get('status')}"
        if not isinstance(t.get("elapsed_ms"), (int, float)):
            return False, f"{t['agent']} missing elapsed_ms"
    return True, "all agents succeeded with timing"


def _check_required_fields(state) -> tuple[bool, str]:
    missing = []
    for f in REQUIRED_FIELDS:
        v = getattr(state, f, None)
        if v is None or (isinstance(v, (list, str)) and len(v) == 0):
            missing.append(f)
    if missing:
        return False, f"missing/empty fields: {missing}"
    return True, "all required fields populated"


def _check_decision_consistency(state) -> tuple[bool, str]:
    prob = state.default_probability
    thr  = state.fairness_threshold_used
    if prob is None or thr is None:
        return False, "probability or threshold missing"
    expected = "REJECTED" if prob >= thr else "APPROVED"
    if state.decision != expected:
        return False, (f"decision={state.decision} but prob={prob:.3f} "
                       f"vs threshold={thr:.3f} → expected={expected}")
    return True, f"decision consistent (prob={prob:.3f} vs thr={thr:.3f})"


def _check_determinism(orch: Orchestrator, borrower_id: int,
                       model: str) -> tuple[bool, str]:
    s1 = orch.run(borrower_id=borrower_id, model=model, verbose=False)
    s2 = orch.run(borrower_id=borrower_id, model=model, verbose=False)
    if s1.default_probability != s2.default_probability:
        return False, (f"non-deterministic probability: "
                       f"{s1.default_probability} vs {s2.default_probability}")
    if s1.decision != s2.decision:
        return False, f"non-deterministic decision: {s1.decision} vs {s2.decision}"
    return True, "deterministic across two runs"


def _check_model_differs(orch: Orchestrator, ids: list[int]) -> tuple[bool, str]:
    """
    For at least one borrower, LR and RF should yield different outputs
    (different probability or different top-factor ordering). If they are
    identical across every borrower, something is wrong.
    """
    any_diff = False
    for bid in ids:
        rf_state = orch.run(borrower_id=bid, model="rf", verbose=False)
        lr_state = orch.run(borrower_id=bid, model="lr", verbose=False)
        if rf_state.default_probability != lr_state.default_probability:
            any_diff = True
            break
    if not any_diff:
        return False, "LR and RF returned identical probabilities for every borrower"
    return True, "LR vs RF produce distinct outputs"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _run_one(orch: Orchestrator, borrower_id: int, model: str) -> dict:
    state = orch.run(borrower_id=borrower_id, model=model, verbose=False)

    per_borrower_checks = [
        ("no_errors",             _check_no_errors(state)),
        ("agent_order",           _check_agent_order(state)),
        ("agent_success_timing",  _check_agent_success(state)),
        ("required_fields",       _check_required_fields(state)),
        ("decision_consistency",  _check_decision_consistency(state)),
    ]
    checks = {name: {"passed": ok, "detail": msg}
              for name, (ok, msg) in per_borrower_checks}
    passed = all(c["passed"] for c in checks.values())

    return {
        "borrower_id":     borrower_id,
        "model":           model,
        "decision":        state.decision,
        "probability":     state.default_probability,
        "risk_level":      state.risk_level,
        "threshold_used":  state.fairness_threshold_used,
        "trace":           state.agent_trace,
        "checks":          checks,
        "passed":          passed,
    }


def run(borrower_ids: list[int] | None = None, verbose: bool = True) -> dict:
    if borrower_ids is None:
        borrower_ids = DEFAULT_SAMPLE_IDS

    if verbose:
        print("=" * 60)
        print(f" Pipeline Validation — Orchestrator Smoke Test "
              f"({len(borrower_ids)} borrowers × 2 models)")
        print("=" * 60)

    orch = Orchestrator()

    borrower_results: list[dict] = []
    for bid in borrower_ids:
        for model in ("rf", "lr"):
            r = _run_one(orch, bid, model)
            borrower_results.append(r)
            if verbose:
                icon = "✓" if r["passed"] else "✗"
                prob = r["probability"]
                prob_s = f"{prob:.3f}" if isinstance(prob, (int, float)) else "N/A"
                dec_s = r["decision"] if r["decision"] is not None else "N/A"
                print(f"  {icon}  #{bid:<4} [{model}]  "
                      f"decision={dec_s:<9}  prob={prob_s}")

    # Global-level checks
    det_ok, det_msg = _check_determinism(orch, borrower_ids[0], "rf")
    diff_ok, diff_msg = _check_model_differs(orch, borrower_ids)

    global_checks = {
        "determinism":          {"passed": det_ok,  "detail": det_msg},
        "model_differentiation": {"passed": diff_ok, "detail": diff_msg},
    }

    all_borrowers_ok = all(r["passed"] for r in borrower_results)
    all_global_ok   = all(c["passed"] for c in global_checks.values())
    overall = all_borrowers_ok and all_global_ok

    result = {
        "borrower_ids":    borrower_ids,
        "models_tested":   ["rf", "lr"],
        "borrower_results": borrower_results,
        "global_checks":    global_checks,
        "overall_passed":   overall,
    }

    os.makedirs(REPORT_DIR, exist_ok=True)
    out_path = os.path.join(REPORT_DIR, "pipeline_validation.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    if verbose:
        print("\n  Global checks:")
        for name, c in global_checks.items():
            print(f"    {'✓' if c['passed'] else '✗'}  {name:<22} {c['detail']}")
        print(f"\n  Report → {out_path}")
        print(f"  Overall: {'PASS' if overall else 'FAIL'}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    p = argparse.ArgumentParser(description="Orchestrator smoke tests")
    p.add_argument("--ids", nargs="+", type=int, default=None,
                   help="borrower indices to test (default: 1 42 100 250 500 750)")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    res = run(borrower_ids=args.ids, verbose=not args.quiet)
    sys.exit(0 if res["overall_passed"] else 1)


if __name__ == "__main__":
    _cli()
