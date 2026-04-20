"""
Run all four validation suites and write a combined report.

Usage
-----
    python -m src.validation.run_all
    python -m src.validation.run_all --skip cross           # skip CV (slow)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from src.validation import (
    cross_validation,
    explainability_validation,
    fairness_validation,
    pipeline_validation,
)

REPORT_DIR = "outputs/reports/validation"


SUITES = {
    "cross":          ("Cross-Validation (Lending Club)",     cross_validation.run),
    "fairness":       ("Fairness (German Credit)",            fairness_validation.run),
    "explainability": ("Explainability (SHAP vs domain)",     explainability_validation.run),
    "pipeline":       ("Pipeline (Orchestrator Smoke Tests)", pipeline_validation.run),
}


def run(skip: list[str] | None = None) -> dict:
    skip = skip or []
    print("\n" + "#" * 64)
    print(" SAAi — Full Validation Run")
    print("#" * 64)

    results: dict = {}
    for key, (label, fn) in SUITES.items():
        if key in skip:
            print(f"\n[SKIPPED] {label}")
            continue
        print(f"\n>>> {label}")
        t0 = time.time()
        try:
            results[key] = {"result": fn(verbose=True), "elapsed_s": None}
            results[key]["elapsed_s"] = round(time.time() - t0, 2)
        except Exception as e:
            results[key] = {
                "result":    {"overall_passed": False, "error": str(e)},
                "elapsed_s": round(time.time() - t0, 2),
            }
            print(f"   !! {label} raised: {e}")

    # Combined summary
    os.makedirs(REPORT_DIR, exist_ok=True)
    combined_path = os.path.join(REPORT_DIR, "combined_validation_report.json")
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 64)
    print(" Validation Summary")
    print("=" * 64)
    for key, (label, _) in SUITES.items():
        if key in skip:
            print(f"  [SKIPPED] {label}")
            continue
        r = results[key]["result"]
        ok = r.get("overall_passed", False)
        print(f"  {'PASS' if ok else 'FAIL':<6} {label}  "
              f"({results[key]['elapsed_s']:.2f}s)")

    print(f"\n  Combined report → {combined_path}")
    overall = all(
        results[k]["result"].get("overall_passed", False)
        for k in results
    )
    print(f"  Overall: {'PASS' if overall else 'FAIL'}\n")
    return {"suites": results, "overall_passed": overall}


def _cli():
    p = argparse.ArgumentParser(description="Run all validation suites")
    p.add_argument("--skip", nargs="*", default=[],
                   choices=list(SUITES.keys()),
                   help="names of suites to skip")
    args = p.parse_args()
    res = run(skip=args.skip)
    sys.exit(0 if res["overall_passed"] else 1)


if __name__ == "__main__":
    _cli()
