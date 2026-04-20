"""
Data Retrieval Agent — Task 6.

Responsibility: fetch a borrower's profile from the data store given a
`borrower_id`. In this version we support:
  • "german_credit_csv" — the 250-row test set (default)
  • "plaid"             — fallback (stub) using MCP Plaid tools

Writes to state:
  - data_source
  - raw_profile            (dict of pre-encoding feature values)
  - feature_names          (list matching the encoded model input order)
  - actual_label           (ground truth, if available)
"""

from __future__ import annotations

from src.agents.base import AgentState, BaseAgent
from src.agents.context import get_context


class DataRetrievalAgent(BaseAgent):
    name = "data_retrieval_agent"
    description = "Fetches borrower profile (German Credit dataset or Plaid)."

    def __init__(self, source: str = "german_credit_csv"):
        self.source = source

    # ------------------------------------------------------------------
    def run(self, state: AgentState) -> AgentState:
        if self.source == "german_credit_csv":
            return self._from_local_csv(state)
        elif self.source == "plaid":
            return self._from_plaid(state)
        else:
            raise ValueError(f"Unknown data source: {self.source!r}")

    # ------------------------------------------------------------------
    def _from_local_csv(self, state: AgentState) -> AgentState:
        """Pull the borrower from the cached test set."""
        if state.borrower_id is None:
            raise ValueError("borrower_id is required for german_credit_csv source")

        ctx = get_context()
        raw_test = ctx["raw_test"]
        y_test   = ctx["y_test"]
        fn       = ctx["feature_names"]

        n = len(raw_test)
        if not (0 <= state.borrower_id < n):
            raise IndexError(
                f"borrower_id={state.borrower_id} out of range (0..{n - 1})"
            )

        row = raw_test.iloc[state.borrower_id]
        state.data_source   = "german_credit_csv"
        state.raw_profile   = {k: _to_py(v) for k, v in row.items()}
        state.feature_names = list(fn)
        state.actual_label  = int(y_test.iloc[state.borrower_id])
        return state

    # ------------------------------------------------------------------
    def _from_plaid(self, state: AgentState) -> AgentState:
        """
        Stub: in production this would call the MCP Plaid tools to build a
        borrower profile from bank transactions + income. For the course
        demo we raise NotImplementedError — the orchestrator catches it and
        records the failure in the trace.
        """
        raise NotImplementedError(
            "Plaid-backed data retrieval is not yet wired end-to-end. "
            "Use source='german_credit_csv' for the demo."
        )


# ------------------------------------------------------------------
def _to_py(v):
    """Convert numpy / pandas scalar types to native python for JSON safety."""
    try:
        import numpy as np
        if isinstance(v, (np.integer,)):   return int(v)
        if isinstance(v, (np.floating,)):  return float(v)
        if isinstance(v, (np.bool_,)):     return bool(v)
    except ImportError:
        pass
    return v
