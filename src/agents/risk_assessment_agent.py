"""
Risk Assessment Agent — Task 6.

Responsibility: using the borrower profile retrieved by DataRetrievalAgent,
run the trained model and produce a default probability + approval decision.

Uses the production prediction path:
  • Pre-computed model probabilities (from `context`)
  • Fairness thresholds from `outputs/models/fairness_thresholds.json`
    applied via `src.models.bias_mitigation.predict_fair`

Writes to state:
  - default_probability
  - decision               ("APPROVED" | "REJECTED")
  - risk_level
  - fairness_threshold_used
  - group_for_threshold
"""

from __future__ import annotations

import json
import os

from src.agents.base import AgentState, BaseAgent
from src.agents.context import get_context


MODEL_DIR       = "outputs/models"
THRESHOLD_FILE  = "fairness_thresholds.json"


def _risk_level(prob: float) -> str:
    if prob >= 0.75: return "Very High Risk"
    if prob >= 0.55: return "High Risk"
    if prob >= 0.40: return "Moderate Risk"
    if prob >= 0.25: return "Low Risk"
    return "Very Low Risk"


class RiskAssessmentAgent(BaseAgent):
    name = "risk_assessment_agent"
    description = (
        "Predicts default probability and decision using the trained "
        "classifier + fairness-aware threshold."
    )

    # ------------------------------------------------------------------
    def run(self, state: AgentState) -> AgentState:
        if not state.raw_profile:
            raise ValueError(
                "raw_profile missing — DataRetrievalAgent must run first."
            )
        if state.borrower_id is None:
            raise ValueError("borrower_id missing")

        ctx      = get_context()
        model_key = state.model_used
        if model_key not in ("rf", "lr"):
            raise ValueError(f"model_used must be 'rf' or 'lr', got {model_key!r}")

        m      = ctx[model_key]
        probs  = m["probs"]
        preds  = m["preds"]
        prob   = float(probs[state.borrower_id])
        pred   = int(preds[state.borrower_id])

        # Look up which fairness threshold applied to this borrower (by group)
        threshold, group = self._lookup_threshold(state, model_key)

        state.default_probability      = round(prob, 4)
        state.decision                 = "REJECTED" if pred == 1 else "APPROVED"
        state.risk_level               = _risk_level(prob)
        state.fairness_threshold_used  = threshold
        state.group_for_threshold      = group
        return state

    # ------------------------------------------------------------------
    def _lookup_threshold(self, state: AgentState, model_key: str):
        """Return (threshold, group_name) actually used for this borrower."""
        path = os.path.join(MODEL_DIR, THRESHOLD_FILE)
        if not os.path.exists(path):
            return None, None

        with open(path) as f:
            thresholds = json.load(f)

        # File schema: {model: {protected_attr: {group: threshold, ...}}}
        model_name = "random_forest" if model_key == "rf" else "logistic_regression"
        if model_name not in thresholds:
            return None, None

        # Prefer sex grouping; fall back to age
        sex_val = state.raw_profile.get("personal_status_sex")
        model_thr = thresholds[model_name]

        for attr in ("personal_status_sex", "age"):
            if attr not in model_thr:
                continue
            groups = model_thr[attr]
            if attr == "personal_status_sex" and sex_val is not None:
                t = groups.get(str(sex_val))
                if t is not None:
                    return float(t), f"{attr}={sex_val}"
            # fallback: return first group entry
            first = next(iter(groups.items()), None)
            if first:
                return float(first[1]), f"{attr}={first[0]}"

        return None, None
