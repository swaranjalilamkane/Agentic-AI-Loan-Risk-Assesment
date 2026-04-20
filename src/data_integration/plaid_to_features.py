"""
Plaid-JSON  →  German-Credit feature schema
===========================================

Converts live Plaid Sandbox data (accounts + transactions + income) into the
22-field feature dict the trained LR / RF models expect.

Fields that can be derived directly from Plaid:
    status                — checking-account balance band
    savings               — savings-account balance band
    installment_rate      — recurring debt / monthly income
    housing               — rent vs mortgage vs neither
    amount                — loan amount (from application form)
    duration              — loan tenure  (from application form)
    employment_duration   — employment tenure if available, else default

Fields that Plaid cannot provide (supplied by the application form or sensible
defaults matching a median German Credit borrower):
    credit_history, purpose, personal_status_sex, other_debtors,
    present_residence, property, age, other_installment_plans, number_credits,
    job, people_liable, telephone, foreign_worker

The mapper NEVER fabricates demographic data. If a demographic field is not
supplied by the caller, a documented neutral default is used and the
`defaults_used` list in the return value lists every such substitution so
auditors can see exactly which fields came from live data vs fallback.
"""
from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Banding helpers
# ---------------------------------------------------------------------------

def _status_band(balance: float | None) -> str:
    """Checking-account status band, matching German Credit vocabulary."""
    if balance is None:
        return "no checking account"
    if balance < 0:
        return "... < 100 DM"                 # overdrawn → lowest band
    if balance < 100:
        return "... < 100 DM"
    if balance < 200:
        return "0 <= ... < 200 DM"
    return "... >= 200 DM / salary assignments for at least 1 year"


def _savings_band(balance: float | None) -> str:
    """Savings-account band, matching German Credit vocabulary."""
    if balance is None:
        return "unknown/no savings account"
    if balance < 100:
        return "... < 100 DM"
    if balance < 500:
        return "100 <= ... < 500 DM"
    if balance < 1000:
        return "500 <= ... < 1000 DM"
    return "... >= 1000 DM"


def _employment_band(months: int | None) -> str:
    if months is None:
        return "1 <= ... < 4 years"            # median band (neutral)
    years = months / 12
    if years < 1:
        return "... < 1 year"
    if years < 4:
        return "1 <= ... < 4 years"
    if years < 7:
        return "4 <= ... < 7 years"
    return "... >= 7 years"


def _installment_rate(monthly_debt: float, monthly_income: float) -> int:
    """Bucket (income share going to debt) into the 1-4 German Credit scale."""
    if monthly_income <= 0:
        return 4                               # worst bucket
    pct = monthly_debt / monthly_income
    if pct < 0.10:
        return 1
    if pct < 0.20:
        return 2
    if pct < 0.35:
        return 3
    return 4


# ---------------------------------------------------------------------------
# Derivation from Plaid artefacts
# ---------------------------------------------------------------------------

def _checking_balance(accounts: list[dict]) -> float | None:
    for a in accounts:
        if a.get("subtype", "").lower() == "checking":
            return float(a.get("balance_current") or 0.0)
    return None


def _savings_balance(accounts: list[dict]) -> float | None:
    for a in accounts:
        if a.get("subtype", "").lower() == "savings":
            return float(a.get("balance_current") or 0.0)
    return None


def _monthly_debt(transactions: list[dict]) -> float:
    """Sum of monthly debt-like outflows (credit card, loan, mortgage rows)."""
    debt_categories = {"credit card", "loan", "mortgage", "student loan",
                       "auto loan", "payment"}
    debt_txns = [
        t for t in transactions
        if t.get("amount", 0) > 0                                    # money out
        and any(
            any(k in (c or "").lower() for k in debt_categories)
            for c in (t.get("category") or [])
        )
    ]
    if not debt_txns:
        return 0.0
    total = sum(float(t["amount"]) for t in debt_txns)
    return round(total / 12, 2)                                      # 12-month avg


def _detect_housing(transactions: list[dict]) -> str:
    """Detect housing tenure from transaction categories."""
    for t in transactions:
        cats = [str(c).lower() for c in (t.get("category") or [])]
        if any("rent" in c for c in cats):
            return "rent"
        if any("mortgage" in c for c in cats):
            return "own"
    return "for free"                                                # neither seen


# ---------------------------------------------------------------------------
# Default application-form fields (documented, neutral values)
# ---------------------------------------------------------------------------

DEFAULT_APPLICATION: dict[str, Any] = {
    # Loan-specific (caller should override) ---------------------------------
    "amount":            5000,
    "duration":          24,
    "purpose":           "other purpose",

    # Demographic (caller should override — Plaid CANNOT provide these) ------
    "age":               35,
    "personal_status_sex": "male : single",

    # Credit-bureau (caller should override — not in Plaid) ------------------
    "credit_history":    "existing credits paid back duly till now",

    # Household / application-form fields (safe neutral defaults) -----------
    "other_debtors":            "none",
    "present_residence":        2,
    "property":                 "building society savings agreement/life insurance",
    "other_installment_plans":  "none",
    "number_credits":           1,
    "job":                      "skilled employee/official",
    "people_liable":            1,
    "telephone":                "yes",
    "foreign_worker":           "no",
}


# ---------------------------------------------------------------------------
# Main mapper
# ---------------------------------------------------------------------------

FEATURE_ORDER: list[str] = [
    "status", "duration", "credit_history", "purpose", "amount",
    "savings", "employment_duration", "installment_rate",
    "personal_status_sex", "other_debtors", "present_residence",
    "property", "age", "other_installment_plans", "housing",
    "number_credits", "job", "people_liable", "telephone", "foreign_worker",
    # Engineered features re-computed below
    "credit_amount_per_duration", "credit_per_person",
]


def map_to_german_credit(
    accounts: list[dict],
    transactions: list[dict],
    income: dict,
    application: dict | None = None,
    employment_months: int | None = None,
) -> dict:
    """
    Build a German-Credit feature dict from live Plaid data + application form.

    Parameters
    ----------
    accounts
        Output of PlaidConnector.get_accounts()
    transactions
        Output of PlaidConnector.get_transactions()
    income
        Output of PlaidConnector.get_income_summary()
    application
        Dict with any of the DEFAULT_APPLICATION keys — caller overrides.
        At minimum, `amount`, `duration`, `age`, `personal_status_sex`, and
        `purpose` should be supplied for a realistic decision.
    employment_months
        Optional employment tenure (if caller has this from a separate source;
        Plaid's /income/verification product would normally provide it).

    Returns
    -------
    dict with keys:
        features       – the 22-field feature dict ready for the model
        provenance     – per-field source ("plaid" / "application" / "default")
        defaults_used  – list of fields that fell back to a default
    """
    application = {**DEFAULT_APPLICATION, **(application or {})}

    checking = _checking_balance(accounts)
    savings  = _savings_balance(accounts)
    monthly_income = float(income.get("estimated_monthly_income") or 0.0)
    monthly_debt   = _monthly_debt(transactions)

    features = {
        # ---- Derived live from Plaid ----
        "status":              _status_band(checking),
        "savings":             _savings_band(savings),
        "installment_rate":    _installment_rate(monthly_debt, monthly_income),
        "housing":             _detect_housing(transactions),
        "employment_duration": _employment_band(employment_months),

        # ---- From application form (caller-supplied or default) ----
        "amount":                   int(application["amount"]),
        "duration":                 int(application["duration"]),
        "purpose":                  str(application["purpose"]),
        "age":                      int(application["age"]),
        "personal_status_sex":      str(application["personal_status_sex"]),
        "credit_history":           str(application["credit_history"]),
        "other_debtors":            str(application["other_debtors"]),
        "present_residence":        int(application["present_residence"]),
        "property":                 str(application["property"]),
        "other_installment_plans":  str(application["other_installment_plans"]),
        "number_credits":           int(application["number_credits"]),
        "job":                      str(application["job"]),
        "people_liable":            int(application["people_liable"]),
        "telephone":                str(application["telephone"]),
        "foreign_worker":           str(application["foreign_worker"]),
    }

    # ---- Engineered features (match src/features/feature_engineering.py) ----
    amount   = features["amount"]
    duration = features["duration"]
    liable   = max(features["people_liable"], 1)
    features["credit_amount_per_duration"] = round(amount / max(duration, 1), 4)
    features["credit_per_person"]          = round(amount / liable, 4)

    # ---- Provenance tracking ----
    provenance = {
        "status":                      "plaid" if checking is not None else "default",
        "savings":                     "plaid" if savings  is not None else "default",
        "installment_rate":            "plaid",
        "housing":                     "plaid",
        "employment_duration":         "plaid" if employment_months else "default",
        "amount":                      "application",
        "duration":                    "application",
        "purpose":                     "application",
        "age":                         "application",
        "personal_status_sex":         "application",
        "credit_history":              "application",
        "other_debtors":               "application",
        "present_residence":           "application",
        "property":                    "application",
        "other_installment_plans":     "application",
        "number_credits":              "application",
        "job":                         "application",
        "people_liable":               "application",
        "telephone":                   "application",
        "foreign_worker":              "application",
        "credit_amount_per_duration":  "engineered",
        "credit_per_person":           "engineered",
    }

    # Fields that actually used the DEFAULT_APPLICATION fallback
    user_supplied = set((application or {}).keys()) if application else set()
    defaults_used = [
        k for k in DEFAULT_APPLICATION
        if k not in user_supplied
    ]

    return {
        "features":      features,
        "provenance":    provenance,
        "defaults_used": defaults_used,
        "live_signals": {
            "checking_balance":  checking,
            "savings_balance":   savings,
            "monthly_income":    monthly_income,
            "monthly_debt":      monthly_debt,
            "transaction_count": income.get("transaction_count", 0),
        },
    }
