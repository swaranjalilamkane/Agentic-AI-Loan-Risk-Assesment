"""
Multi-agent orchestrator for the Loan Risk Assessment system (Task 6).

Pipeline:
    Data Retrieval Agent  →  Risk Assessment Agent  →  Explanation Generator Agent

Each agent operates on a shared `AgentState` object. The orchestrator drives the
sequence and captures a trace of every step for transparency / debugging.
"""

from src.agents.base import AgentState, BaseAgent
from src.agents.data_retrieval_agent import DataRetrievalAgent
from src.agents.risk_assessment_agent import RiskAssessmentAgent
from src.agents.explanation_agent import ExplanationAgent
from src.agents.orchestrator import Orchestrator

__all__ = [
    "AgentState",
    "BaseAgent",
    "DataRetrievalAgent",
    "RiskAssessmentAgent",
    "ExplanationAgent",
    "Orchestrator",
]
