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

    def __init__(
        self,
        source: str = "german_credit_csv",
        access_token: str | None = None,
        application: dict | None = None,
        employment_months: int | None = None,
        institution_id: str | None = None,
    ):
        """
        Parameters
        ----------
        source              "german_credit_csv" (default) or "plaid"
        access_token        Plaid access token — if omitted, a sandbox token is
                            minted on the fly using SANDBOX credentials.
        application         Application-form dict (loan amount, duration, age,
                            etc.) used by the Plaid mapper to fill fields Plaid
                            cannot provide.
        employment_months   Optional employment tenure in months.
        institution_id      Plaid sandbox institution id (defaults to ins_109508).
        """
        self.source            = source
        self.access_token      = access_token
        self.application       = application or {}
        self.employment_months = employment_months
        self.institution_id    = institution_id or "ins_109508"

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
        Live Plaid path.

        1. Obtain an access token (use provided one, or mint a sandbox token).
        2. Pull accounts + transactions + income summary.
        3. Map the JSON onto the 22-field German-Credit feature schema via
           src.data_integration.plaid_to_features.map_to_german_credit.
        4. Populate state.raw_profile (human-readable) and state.live_features
           (the feature dict that RiskAssessmentAgent will score live).
        """
        from src.data_integration.plaid_connector import PlaidConnector
        from src.data_integration.plaid_to_features import (
            FEATURE_ORDER, map_to_german_credit,
        )

        conn = PlaidConnector()

        # 1. Access token — either pre-supplied, or auto-minted for sandbox.
        access_token = self.access_token
        if not access_token:
            tok = conn.create_sandbox_access_token(institution_id=self.institution_id)
            access_token = tok["access_token"]

        # 2. Pull live data.
        accounts     = conn.get_accounts(access_token)
        transactions = conn.get_transactions(access_token)
        income       = conn.get_income_summary(access_token)

        # 3. Map to the German-Credit feature schema.
        mapping = map_to_german_credit(
            accounts=accounts,
            transactions=transactions,
            income=income,
            application=self.application,
            employment_months=self.employment_months,
        )

        # 4. Populate state.
        state.data_source         = "plaid"
        state.raw_profile         = dict(mapping["features"])  # human-readable copy
        state.feature_names       = list(FEATURE_ORDER)
        state.live_features       = mapping["features"]
        state.feature_provenance  = mapping["provenance"]
        state.actual_label        = None                       # unknown for live borrowers

        # Stash the live signals in raw_profile under reserved keys so the
        # UI / audit log can show what actually came from the bank.
        state.raw_profile["_plaid_live_signals"] = mapping["live_signals"]
        state.raw_profile["_plaid_defaults_used"] = mapping["defaults_used"]
        return state


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
