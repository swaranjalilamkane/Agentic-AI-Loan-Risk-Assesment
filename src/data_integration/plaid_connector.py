from plaid import Client

class PlaidConnector:
    def __init__(self, client_id, secret):
        self.client = Client(
            client_id=client_id,
            secret=secret,
            environment='sandbox'
        )

    def get_transactions(self, access_token):
        response = self.client.Transactions.get(
            access_token,
            start_date="2023-01-01",
            end_date="2023-12-31"
        )
        return response["transactions"]
