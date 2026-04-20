"""
Orchestrator — Task 6.

Runs the three agents in sequence:
    Data Retrieval  →  Risk Assessment  →  Explanation Generator

Usage
-----
    from src.agents import Orchestrator
    orch = Orchestrator()
    state = orch.run(borrower_id=100, model="rf")
    print(state.narrative)

CLI
---
    python -m src.agents.orchestrator 100
    python -m src.agents.orchestrator 100 --lr
    python -m src.agents.orchestrator 100 --save outputs/reports/decision_100.json
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict

from src.agents.base import AgentState, BaseAgent
from src.agents.data_retrieval_agent import DataRetrievalAgent
from src.agents.risk_assessment_agent import RiskAssessmentAgent
from src.agents.explanation_agent import ExplanationAgent


class Orchestrator:
    """Runs a fixed pipeline of agents against a single borrower."""

    def __init__(self, agents: list[BaseAgent] | None = None):
        self.agents: list[BaseAgent] = agents or [
            DataRetrievalAgent(source="german_credit_csv"),
            RiskAssessmentAgent(),
            ExplanationAgent(),
        ]

    # ------------------------------------------------------------------
    def run(
        self,
        borrower_id: int,
        model: str = "rf",
        top_n_factors: int = 5,
        stop_on_error: bool = True,
        verbose: bool = False,
    ) -> AgentState:
        """Kick off the full pipeline for one borrower."""
        state = AgentState(
            borrower_id   = borrower_id,
            model_used    = model,
            top_n_factors = top_n_factors,
        )

        for agent in self.agents:
            if verbose:
                print(f"▶ Running {agent.name} …")
            state = agent(state)

            last = state.agent_trace[-1] if state.agent_trace else {}
            status = last.get("status", "?")
            if verbose:
                icon = "✓" if status == "success" else "✗"
                ms   = last.get("elapsed_ms", "?")
                print(f"   {icon} {agent.name} ({ms} ms)")
                if status != "success":
                    print(f"     error: {last.get('error', '')}")

            if status == "failed" and stop_on_error:
                break

        return state

    # ------------------------------------------------------------------
    @staticmethod
    def print_report(state: AgentState) -> None:
        """Pretty-print the orchestrator trace + final decision."""
        print("\n" + "=" * 64)
        print(" AGENT ORCHESTRATOR — Decision Report")
        print("=" * 64)

        # Trace table
        print("\n Agent Execution Trace:")
        print(" " + "-" * 60)
        print(f" {'Agent':<28}{'Status':<10}{'Time (ms)':>10}")
        print(" " + "-" * 60)
        for entry in state.agent_trace:
            print(
                f" {entry['agent']:<28}"
                f"{entry['status']:<10}"
                f"{entry.get('elapsed_ms', 0):>10.1f}"
            )

        if state.errors:
            print("\n  Errors:")
            for e in state.errors:
                print(f"  - {e}")
            return

        # Final decision
        print("\n Final Decision:")
        print(" " + "-" * 60)
        print(f"   Borrower ID          : #{state.borrower_id}")
        print(f"   Model                : {state.model_used.upper()}")
        print(f"   Data source          : {state.data_source}")
        print(f"   Default probability  : {state.default_probability:.1%}")
        print(f"   Risk level           : {state.risk_level}")
        print(f"   Decision             : {state.decision}")
        if state.fairness_threshold_used is not None:
            print(
                f"   Fairness threshold   : "
                f"{state.fairness_threshold_used:.3f}  "
                f"({state.group_for_threshold})"
            )
        if state.actual_label is not None:
            actual = "Default" if state.actual_label == 1 else "Good Credit"
            print(f"   Ground truth         : {actual}")

        # Narrative
        if state.narrative:
            print("\n Explanation:")
            print(" " + "-" * 60)
            for line in state.narrative.splitlines():
                print(f"   {line}")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _save_state(state: AgentState, path: str) -> None:
    """Serialize state to JSON."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(state), f, indent=2, default=str)
    print(f"\n  Saved → {path}")


def main() -> None:
    args = sys.argv[1:]
    if not args or not args[0].lstrip("-").isdigit():
        print("Usage: python -m src.agents.orchestrator <borrower_id> "
              "[--lr] [--save PATH]")
        sys.exit(1)

    borrower_id = int(args[0])
    model       = "lr" if "--lr" in args else "rf"
    save_path   = None
    if "--save" in args:
        i = args.index("--save")
        if i + 1 < len(args):
            save_path = args[i + 1]

    orch  = Orchestrator()
    state = orch.run(borrower_id=borrower_id, model=model, verbose=True)
    Orchestrator.print_report(state)

    if save_path:
        _save_state(state, save_path)


if __name__ == "__main__":
    main()
