"""
MCP-style tool wrappers around PlaidConnector.

Each class follows the tool-call contract expected by the multi-agent
orchestrator:

  tool.name   -> str  (unique identifier sent in LLM tool_use blocks)
  tool.schema -> dict (JSON-Schema for the ``input`` field)
  tool(input) -> dict (JSON-serialisable result)

The connector is instantiated lazily so that missing env vars only surface
when a tool is actually invoked.
"""

from __future__ import annotations

import os
from typing import Any


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class _PlaidBaseTool:
    """Shared lazy-init logic for all Plaid tools."""

    name: str = ""
    schema: dict = {}

    def _connector(self):
        from src.data_integration.plaid_connector import PlaidConnector
        return PlaidConnector(
            client_id=os.getenv("PLAID_CLIENT_ID", ""),
            secret=os.getenv("PLAID_SECRET", ""),
            environment=os.getenv("PLAID_ENV", "sandbox"),
        )

    def __call__(self, input: dict[str, Any]) -> dict:  # noqa: A002
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

class CreateLinkTokenTool(_PlaidBaseTool):
    """Generate a Plaid Link token for a user (step 1 of the OAuth flow)."""

    name = "plaid_create_link_token"
    schema = {
        "type": "object",
        "properties": {
            "user_id": {
                "type": "string",
                "description": "Unique identifier for the applicant in our system.",
            }
        },
        "required": ["user_id"],
    }

    def __call__(self, input: dict) -> dict:
        return self._connector().create_link_token(input["user_id"])


class ExchangePublicTokenTool(_PlaidBaseTool):
    """Exchange the public token from Plaid Link for a durable access token."""

    name = "plaid_exchange_public_token"
    schema = {
        "type": "object",
        "properties": {
            "public_token": {
                "type": "string",
                "description": "Short-lived public token returned by Plaid Link.",
            }
        },
        "required": ["public_token"],
    }

    def __call__(self, input: dict) -> dict:
        return self._connector().exchange_public_token(input["public_token"])


class GetAccountsTool(_PlaidBaseTool):
    """Retrieve all bank accounts linked to a Plaid access token."""

    name = "plaid_get_accounts"
    schema = {
        "type": "object",
        "properties": {
            "access_token": {
                "type": "string",
                "description": "Plaid access token for the applicant's institution.",
            }
        },
        "required": ["access_token"],
    }

    def __call__(self, input: dict) -> dict:
        accounts = self._connector().get_accounts(input["access_token"])
        return {"accounts": accounts, "count": len(accounts)}


class GetTransactionsTool(_PlaidBaseTool):
    """Retrieve up to 500 transactions for a date range (defaults: trailing 12 months)."""

    name = "plaid_get_transactions"
    schema = {
        "type": "object",
        "properties": {
            "access_token": {
                "type": "string",
                "description": "Plaid access token for the applicant's institution.",
            },
            "start_date": {
                "type": "string",
                "format": "date",
                "description": "ISO-8601 start date (YYYY-MM-DD). Optional.",
            },
            "end_date": {
                "type": "string",
                "format": "date",
                "description": "ISO-8601 end date (YYYY-MM-DD). Optional.",
            },
        },
        "required": ["access_token"],
    }

    def __call__(self, input: dict) -> dict:
        import datetime

        start = (
            datetime.date.fromisoformat(input["start_date"])
            if "start_date" in input
            else None
        )
        end = (
            datetime.date.fromisoformat(input["end_date"])
            if "end_date" in input
            else None
        )
        txns = self._connector().get_transactions(
            input["access_token"], start_date=start, end_date=end
        )
        return {"transactions": txns, "count": len(txns)}


class GetIncomeSummaryTool(_PlaidBaseTool):
    """Derive an estimated monthly income and spend summary from transactions."""

    name = "plaid_get_income_summary"
    schema = {
        "type": "object",
        "properties": {
            "access_token": {
                "type": "string",
                "description": "Plaid access token for the applicant's institution.",
            }
        },
        "required": ["access_token"],
    }

    def __call__(self, input: dict) -> dict:
        return self._connector().get_income_summary(input["access_token"])


class CreateSandboxAccessTokenTool(_PlaidBaseTool):
    """
    Sandbox-only: create an access token directly without going through
    Plaid Link UI. Uses the default sandbox credentials (user_good / pass_good).
    """

    name = "plaid_create_sandbox_access_token"
    schema = {
        "type": "object",
        "properties": {
            "institution_id": {
                "type": "string",
                "description": (
                    "Plaid institution ID. Defaults to ins_109508 (First Platypus Bank). "
                    "Other options: ins_3 (Chase), ins_4 (Wells Fargo), ins_5 (BofA), ins_6 (Citi)."
                ),
            },
            "products": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Plaid products to enable (e.g. ['transactions', 'auth']). Defaults to ['transactions'].",
            },
        },
        "required": [],
    }

    def __call__(self, input: dict) -> dict:
        return self._connector().create_sandbox_access_token(
            institution_id=input.get("institution_id", "ins_109508"),
            products=input.get("products", ["transactions"]),
        )
