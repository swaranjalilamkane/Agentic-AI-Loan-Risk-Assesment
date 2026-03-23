"""
MCP-style tool definitions for the Loan Risk Assessment system.

Each tool is a plain callable with:
  - A ``name``   attribute (str)  – unique tool identifier
  - A ``schema`` attribute (dict) – JSON-Schema describing the input parameters
  - A ``__call__``                – executes the tool and returns a JSON-serialisable dict

The orchestrator (Task 6) discovers tools via ``ALL_TOOLS`` and dispatches calls
by matching the tool name emitted by the LLM.
"""

from src.mcp_tools.plaid_tools import (
    CreateLinkTokenTool,
    ExchangePublicTokenTool,
    GetAccountsTool,
    GetTransactionsTool,
    GetIncomeSummaryTool,
    CreateSandboxAccessTokenTool,
)

ALL_TOOLS = [
    CreateLinkTokenTool(),
    ExchangePublicTokenTool(),
    GetAccountsTool(),
    GetTransactionsTool(),
    GetIncomeSummaryTool(),
    CreateSandboxAccessTokenTool(),   # sandbox only
]

__all__ = ["ALL_TOOLS"]
