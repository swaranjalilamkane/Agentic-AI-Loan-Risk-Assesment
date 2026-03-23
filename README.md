# Agentic AI Loan Risk Assessment

An Agentic AI Framework for Multi-Source Loan Risk Assessment and Fair Lending Decisions.

---

## Project Objectives

| # | Objective | Status |
|---|-----------|--------|
| 1 | Multi-agent architecture (Data Retrieval, Risk Assessment, Explanation Generator) | In Progress |
| 2 | Heterogeneous data integration via MCP (Plaid + datasets) | Done |
| 3 | Fair and bias-aware credit scoring models | Done |
| 4 | Explainable AI with SHAP | Planned |
| 5 | RAG system for policy compliance | Planned |
| 6 | Full evaluation (accuracy, fairness, efficiency) | Planned |

---

## Setup

### 1. Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in your PLAID_CLIENT_ID and PLAID_SECRET
```

### 3. Add datasets

Download and place in `data/raw/`:
- [Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club) → `data/raw/lendingclub.csv`
- [German Credit Dataset](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)) → `data/raw/german_credit.csv`

### 4. Run the full pipeline

```bash
python run_pipeline.py
```

---

## Project Progress

### Week 1: Project Planning

- Identified the loan risk prediction problem
- Selected datasets (German Credit, Lending Club)
- Planned multi-agent architecture
- Created repository structure and project roadmap

### Week 2: Architecture Design

- Defined data integration and processing workflow
- Designed feature engineering and borrower profiling modules
- Planned risk scoring and explanation components
- Prepared module structure for implementation

### Week 3: Data Integration Pipeline

Implemented a full data integration pipeline:

- Ingests German Credit and Lending Club datasets
- Cleans and standardizes borrower data
- Performs feature engineering (age group, credit intensity, per-capita liability)
- Generates borrower profile JSON objects (`outputs/borrower_profiles/`)

### Week 4: Plaid MCP Connector + Credit Risk Models

#### Plaid Connector (`src/data_integration/plaid_connector.py`)

Updated to the modern `plaid-python` v8+ API. Supports the full Sandbox flow:

| Method | Description |
|--------|-------------|
| `create_link_token(user_id)` | Start the Plaid Link OAuth flow |
| `exchange_public_token(token)` | Get a durable access token |
| `create_sandbox_access_token()` | Bypass Link UI in Sandbox (uses `user_good` / `pass_good`) |
| `get_accounts(access_token)` | Fetch linked accounts and balances |
| `get_transactions(access_token)` | Retrieve up to 500 transactions |
| `get_income_summary(access_token)` | Estimated monthly income and spend |

Quick sandbox test (no UI needed):

```python
from src.data_integration.plaid_connector import PlaidConnector

conn = PlaidConnector()                          # reads .env
result = conn.create_sandbox_access_token()      # user_good / pass_good
access_token = result["access_token"]

print(conn.get_accounts(access_token))
print(conn.get_income_summary(access_token))
```

#### MCP Tools Layer (`src/mcp_tools/`)

MCP-style tool wrappers for the multi-agent orchestrator. Each tool exposes `name`, `schema` (JSON Schema), and `__call__`.

| Tool name | Description |
|-----------|-------------|
| `plaid_create_link_token` | Generate a Link token |
| `plaid_exchange_public_token` | Exchange for access token |
| `plaid_create_sandbox_access_token` | Sandbox shortcut |
| `plaid_get_accounts` | Fetch accounts |
| `plaid_get_transactions` | Fetch transactions |
| `plaid_get_income_summary` | Derive income summary |

#### Credit Risk Models (`src/models/`)

Two classifiers trained on the German Credit dataset with `class_weight='balanced'` to handle the 30/70 class imbalance:

- **Logistic Regression** (with `StandardScaler`)
- **Random Forest** (100 estimators, max depth 8)

Target encoding: `credit_risk == 1` (bad) → `1` (default), `credit_risk == 2` (good) → `0`.

Accuracy metrics: **AUC-ROC**, **Precision**, **Recall**

#### Fairness Evaluation (`src/models/fairness.py`)

| Metric | What it measures | Bias threshold |
|--------|-----------------|----------------|
| Demographic Parity Difference | Range of approval rates across groups | > 0.10 |
| Equalized Odds Difference | Max range of TPR / FPR across groups | > 0.10 |

Protected attributes tested:
- `personal_status_sex` (gender × marital status)
- `age` (binarised: ≤ 25 vs > 25)

Run model training and fairness evaluation:

```bash
python -m src.models.evaluate
```

Output saved to `outputs/reports/evaluation_report.json`.

---

## Repository Structure

```
├── data/
│   ├── raw/               # Raw datasets (not committed)
│   └── processed/         # Cleaned CSVs (not committed)
├── outputs/
│   ├── features/          # Engineered features CSV
│   ├── borrower_profiles/ # Per-borrower JSON profiles
│   ├── models/            # Saved .pkl model files
│   └── reports/           # evaluation_report.json
├── src/
│   ├── data_integration/
│   │   ├── datasets.py
│   │   ├── feature_engineering.py
│   │   ├── build_borrower_profile.py
│   │   └── plaid_connector.py
│   ├── mcp_tools/
│   │   ├── __init__.py    # ALL_TOOLS registry
│   │   └── plaid_tools.py
│   ├── models/
│   │   ├── credit_model.py
│   │   ├── fairness.py
│   │   └── evaluate.py
│   └── utils/
│       ├── config.py
│       └── logger.py
├── .env.example
├── requirements.txt
└── run_pipeline.py        # Full pipeline entry point
```

---

## Upcoming

- **Task 5** — SHAP explainability module + RAG pipeline for policy compliance
- **Task 6** — Multi-agent orchestrator (Data Retrieval → Risk Assessment → Explanation Generator)
- **Task 7** — Demo application and full benchmark evaluation report
