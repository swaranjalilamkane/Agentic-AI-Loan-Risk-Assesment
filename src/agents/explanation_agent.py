"""
Explanation Generator Agent — Task 6.

Responsibility: take the model's decision from RiskAssessmentAgent, compute
the borrower's SHAP contributions, and produce a plain-English narrative +
ranked lists of risk / protective factors.

Delegates the heavy lifting to `src.models.explain_borrower.explain_borrower`.

Writes to state:
  - narrative             (multi-line English text)
  - risk_factors          (top-N features pushing toward default)
  - protective_factors    (top-N features pushing toward approval)
  - shap_report           (all features, sorted by |SHAP|)
"""

from __future__ import annotations

import pandas as pd

from src.agents.base import AgentState, BaseAgent
from src.agents.context import get_context
from src.models.explain_borrower import explain_borrower


class ExplanationAgent(BaseAgent):
    name = "explanation_agent"
    description = (
        "Generates SHAP-based human-readable explanation "
        "(risk factors, protective factors, narrative)."
    )

    # ------------------------------------------------------------------
    def run(self, state: AgentState) -> AgentState:
        if state.default_probability is None or state.decision is None:
            raise ValueError(
                "RiskAssessmentAgent must run before ExplanationAgent "
                "(default_probability / decision missing)."
            )

        ctx       = get_context()
        model_key = state.model_used
        m         = ctx[model_key]

        raw_row = pd.Series(state.raw_profile)
        predicted_label = 1 if state.decision == "REJECTED" else 0

        result = explain_borrower(
            shap_values     = m["shap_vals"][state.borrower_id],
            raw_row         = raw_row,
            feature_names   = ctx["feature_names"],
            base_value      = m["base_value"],
            predicted_prob  = state.default_probability,
            predicted_label = predicted_label,
            actual_label    = state.actual_label,
            borrower_id     = state.borrower_id,
            top_n           = state.top_n_factors,
        )

        state.narrative           = result["narrative"]
        state.risk_factors        = result["risk_factors"]
        state.protective_factors  = result["protective_factors"]
        state.shap_report         = sorted(
            result["factors_detail"],
            key=lambda x: abs(x["shap"]),
            reverse=True,
        )
        return state
