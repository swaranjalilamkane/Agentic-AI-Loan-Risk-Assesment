import os
import datetime
import plaid
from plaid.api import plaid_api
from plaid.model.transactions_get_request import TransactionsGetRequest
from plaid.model.transactions_get_request_options import TransactionsGetRequestOptions
from plaid.model.accounts_get_request import AccountsGetRequest
from plaid.model.link_token_create_request import LinkTokenCreateRequest
from plaid.model.link_token_create_request_user import LinkTokenCreateRequestUser
from plaid.model.products import Products
from plaid.model.country_code import CountryCode
from plaid.model.item_public_token_exchange_request import ItemPublicTokenExchangeRequest
from plaid.model.sandbox_public_token_create_request import SandboxPublicTokenCreateRequest
from plaid.model.sandbox_public_token_create_request_options import SandboxPublicTokenCreateRequestOptions

# Plaid Sandbox default credentials (all test institutions)
SANDBOX_USERNAME = "user_good"
SANDBOX_PASSWORD = "pass_good"

# Commonly used Sandbox institution IDs
SANDBOX_INSTITUTIONS = {
    "chase": "ins_3",
    "wells_fargo": "ins_4",
    "bofa": "ins_5",
    "citi": "ins_6",
    "first_platypus": "ins_109508",   # Plaid's dedicated test institution
}


def _build_client(client_id: str, secret: str, environment: str = "sandbox") -> plaid_api.PlaidApi:
    env_map = {
        "sandbox": plaid.Environment.Sandbox,
        "development": plaid.Environment.Development,
        "production": plaid.Environment.Production,
    }
    configuration = plaid.Configuration(
        host=env_map.get(environment, plaid.Environment.Sandbox),
        api_key={
            "clientId": client_id,
            "secret": secret,
        },
    )
    api_client = plaid.ApiClient(configuration)
    return plaid_api.PlaidApi(api_client)


class PlaidConnector:
    """
    MCP-compatible Plaid Sandbox connector.

    Wraps the Plaid API (plaid-python v8+) and exposes methods that
    follow the Model Context Protocol tool-call pattern used by the
    multi-agent orchestrator.
    """

    def __init__(self, client_id: str | None = None, secret: str | None = None, environment: str = "sandbox"):
        client_id = client_id or os.getenv("PLAID_CLIENT_ID", "")
        secret = secret or os.getenv("PLAID_SECRET", "")
        self.client = _build_client(client_id, secret, environment)

    # ------------------------------------------------------------------
    # Link / Auth helpers
    # ------------------------------------------------------------------

    def create_link_token(self, user_id: str) -> dict:
        """Create a Plaid Link token for the given user (sandbox flow)."""
        request = LinkTokenCreateRequest(
            products=[Products("transactions")],
            client_name="Loan Risk Assessor",
            country_codes=[CountryCode("US")],
            language="en",
            user=LinkTokenCreateRequestUser(client_user_id=user_id),
        )
        response = self.client.link_token_create(request)
        return {"link_token": response["link_token"], "expiration": response["expiration"]}

    def exchange_public_token(self, public_token: str) -> dict:
        """Exchange a public token returned by Plaid Link for an access token."""
        request = ItemPublicTokenExchangeRequest(public_token=public_token)
        response = self.client.item_public_token_exchange(request)
        return {
            "access_token": response["access_token"],
            "item_id": response["item_id"],
        }

    def create_sandbox_access_token(
        self,
        institution_id: str = "ins_109508",
        products: list[str] | None = None,
    ) -> dict:
        """
        Bypass the Link UI in Sandbox to obtain an access token directly.

        Uses the Sandbox-only ``/sandbox/public_token/create`` endpoint.
        Default institution is First Platypus Bank (ins_109508), Plaid's
        dedicated test institution. Credentials: user_good / pass_good.

        Parameters
        ----------
        institution_id : Plaid institution ID (see SANDBOX_INSTITUTIONS dict)
        products       : list of product strings, defaults to ['transactions']

        Returns
        -------
        dict with access_token and item_id ready for data retrieval calls
        """
        if products is None:
            products = ["transactions"]

        request = SandboxPublicTokenCreateRequest(
            institution_id=institution_id,
            initial_products=[Products(p) for p in products],
            options=SandboxPublicTokenCreateRequestOptions(
                override_username=SANDBOX_USERNAME,
                override_password=SANDBOX_PASSWORD,
            ),
        )
        response = self.client.sandbox_public_token_create(request)
        return self.exchange_public_token(response["public_token"])

    # ------------------------------------------------------------------
    # Data retrieval
    # ------------------------------------------------------------------

    def get_accounts(self, access_token: str) -> list[dict]:
        """Return all accounts linked to the given access token."""
        request = AccountsGetRequest(access_token=access_token)
        response = self.client.accounts_get(request)
        accounts = []
        for acct in response["accounts"]:
            accounts.append(
                {
                    "account_id": acct["account_id"],
                    "name": acct["name"],
                    "type": str(acct["type"]),
                    "subtype": str(acct.get("subtype", "")),
                    "balance_current": acct["balances"]["current"],
                    "balance_available": acct["balances"]["available"],
                    "currency": acct["balances"]["iso_currency_code"],
                }
            )
        return accounts

    def get_transactions(
        self,
        access_token: str,
        start_date: datetime.date | None = None,
        end_date: datetime.date | None = None,
        max_results: int = 500,
    ) -> list[dict]:
        """
        Retrieve transactions for the given access token.

        Defaults to the full previous calendar year when no dates are supplied.
        """
        if end_date is None:
            end_date = datetime.date.today()
        if start_date is None:
            start_date = end_date.replace(year=end_date.year - 1)

        options = TransactionsGetRequestOptions(count=min(max_results, 500))
        request = TransactionsGetRequest(
            access_token=access_token,
            start_date=start_date,
            end_date=end_date,
            options=options,
        )
        response = self.client.transactions_get(request)

        transactions = []
        for txn in response["transactions"]:
            transactions.append(
                {
                    "transaction_id": txn["transaction_id"],
                    "date": str(txn["date"]),
                    "name": txn["name"],
                    "amount": txn["amount"],
                    "category": txn.get("category", []),
                    "account_id": txn["account_id"],
                    "pending": txn["pending"],
                }
            )
        return transactions

    def get_income_summary(self, access_token: str) -> dict:
        """
        Derive a simple income summary from transactions.

        Returns estimated monthly income (credits > 0) and average
        monthly spend (debits) computed from the trailing 12 months.
        """
        txns = self.get_transactions(access_token)
        credits = [t["amount"] for t in txns if t["amount"] < 0]   # Plaid: negative = money in
        debits = [t["amount"] for t in txns if t["amount"] > 0]     # positive = money out

        monthly_income = round(abs(sum(credits)) / 12, 2) if credits else 0.0
        monthly_spend = round(sum(debits) / 12, 2) if debits else 0.0

        return {
            "estimated_monthly_income": monthly_income,
            "estimated_monthly_spend": monthly_spend,
            "transaction_count": len(txns),
        }
