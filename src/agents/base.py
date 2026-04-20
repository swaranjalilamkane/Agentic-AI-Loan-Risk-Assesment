"""
Base agent contract and shared AgentState.

All agents in the pipeline implement:
    class FooAgent(BaseAgent):
        name        = "foo_agent"
        description = "Does foo."
        def run(self, state: AgentState) -> AgentState:
            ...

The orchestrator calls `agent(state)` which wraps `run()` with timing + error
capture and appends an entry to `state.agent_trace`.
"""

from __future__ import annotations

import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

@dataclass
class AgentState:
    """State object passed through the full agent pipeline."""

    # ── Inputs (set by caller) ──────────────────────────────────────────────
    borrower_id: int | None = None
    model_used: str = "rf"                # "rf" (Random Forest) or "lr"
    top_n_factors: int = 5

    # ── Populated by Data Retrieval Agent ───────────────────────────────────
    data_source: str | None = None        # "german_credit_csv" | "plaid"
    raw_profile: dict[str, Any] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)

    # Live-inference path (set only when data_source == "plaid") —
    # a full 22-field feature dict ready for direct model scoring + a
    # per-field provenance map showing what came from Plaid vs form vs default.
    live_features: dict[str, Any] = field(default_factory=dict)
    feature_provenance: dict[str, str] = field(default_factory=dict)

    # ── Populated by Risk Assessment Agent ──────────────────────────────────
    default_probability: float | None = None
    decision: str | None = None           # "APPROVED" | "REJECTED"
    risk_level: str | None = None
    fairness_threshold_used: float | None = None
    group_for_threshold: str | None = None
    actual_label: int | None = None       # ground truth, if available

    # ── Populated by Explanation Generator Agent ────────────────────────────
    narrative: str | None = None
    risk_factors: list[dict] = field(default_factory=list)
    protective_factors: list[dict] = field(default_factory=list)
    shap_report: list[dict] = field(default_factory=list)  # all features sorted

    # ── Orchestrator trace ──────────────────────────────────────────────────
    agent_trace: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------

class BaseAgent(ABC):
    """Every agent inherits from this class."""

    name: str = "base_agent"
    description: str = "Base agent. Override `run()`."

    @abstractmethod
    def run(self, state: AgentState) -> AgentState:
        """Subclasses implement the real logic here."""
        raise NotImplementedError

    def __call__(self, state: AgentState) -> AgentState:
        """Invoke the agent with timing + error-capture instrumentation."""
        start = time.time()
        try:
            state = self.run(state)
            elapsed_ms = (time.time() - start) * 1000
            state.agent_trace.append({
                "agent":       self.name,
                "status":      "success",
                "elapsed_ms":  round(elapsed_ms, 1),
                "description": self.description,
            })
        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            err = f"{type(e).__name__}: {e}"
            state.errors.append(f"{self.name}: {err}")
            state.agent_trace.append({
                "agent":       self.name,
                "status":      "failed",
                "elapsed_ms":  round(elapsed_ms, 1),
                "error":       err,
                "traceback":   traceback.format_exc(),
            })
        return state
