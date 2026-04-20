# Agentic AI Loan Risk Assessment

An Agentic AI Framework for Multi-Source Loan Risk Assessment and Fair Lending Decisions.

---

## Project Objectives

| # | Objective | Status |
|---|-----------|--------|
| 1 | Multi-agent architecture (Data Retrieval, Risk Assessment, Explanation Generator) | Done |
| 2 | Heterogeneous data integration via MCP (Plaid + datasets) | Done |
| 3 | Fair and bias-aware credit scoring models | Done |
| 4 | Explainable AI with SHAP + human-readable borrower explanations | Done |
| 5 | Interactive Streamlit dashboard UI | Done |
| 6 | RAG system for policy compliance | Planned |
| 7 | Full evaluation (accuracy, fairness, efficiency) | Planned |

---

## Setup

### 1. Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   OR . venv/bin/activate   # Windows: .venv\Scripts\activate
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

Runs all 7 steps: clean → validate → feature-engineer → borrower profiles → train/evaluate with fairness thresholds → SHAP explainability → human-readable borrower explanations.

```bash
python run_pipeline.py
```

### 5. Launch the interactive dashboard

After the pipeline has completed at least once (so models are saved in `outputs/models/`), launch the UI:

```bash
streamlit run app.py
```

Open the URL shown in the terminal (default: http://localhost:8501) to explore borrower-by-borrower SHAP explanations in a browser.

---

## Interactive Dashboard (`app.py`)

A Streamlit-based UI for exploring model decisions and SHAP explanations.

| Section | What it shows |
|---------|---------------|
| Top header (navy bar) | Project title "SAAi – Agentic AI Loan Risk Assessment" |
| Left sidebar | Borrower index input (0–249), model selector (RF/LR), factor-count slider |
| Decision card | Borrower ID, APPROVED / REJECTED badge, model used, ground-truth outcome |
| Gauge chart | Default probability + risk level (Very Low → Very High) |
| Metric cards | Default probability, prediction vs ground truth (True Positive / False Positive / etc.), top risk factor, top protective factor |
| Risk factors (red) | Top N features pushing the applicant toward default, with SHAP bars |
| Protective factors (green) | Top N features pushing toward approval |
| Full SHAP chart | Horizontal bar chart of every feature's SHAP value (red/green) |
| Borrower profile | Raw feature values for the selected applicant |

Change the borrower index or model in the sidebar and the entire dashboard updates instantly. The first page load triggers model + SHAP loading (~10 s); subsequent lookups are instant (cached with `@st.cache_resource`).

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
│   ├── models/            # Saved .pkl model files + fairness_thresholds.json
│   └── reports/           # evaluation_report.json, shap_report.json,
│                          # borrower_explanations_rf.json, shap/ plots
├── src/
│   ├── data_integration/
│   │   ├── datasets.py
│   │   ├── data_validation.py      # schema, completeness, consistency, fairness checks
│   │   ├── feature_engineering.py
│   │   ├── build_borrower_profile.py
│   │   ├── augment_data.py          # SMOTE-style synthetic augmentation
│   │   └── plaid_connector.py
│   ├── mcp_tools/
│   │   ├── __init__.py    # ALL_TOOLS registry
│   │   └── plaid_tools.py
│   ├── models/
│   │   ├── credit_model.py
│   │   ├── fairness.py
│   │   ├── bias_mitigation.py       # post-processing threshold adjustment
│   │   ├── evaluate.py
│   │   ├── shap_explainer.py        # global + local SHAP plots
│   │   └── explain_borrower.py      # SHAP → plain-English narratives
│   ├── agents/                       # Multi-agent orchestrator (Task 6)
│   │   ├── base.py                  # BaseAgent + AgentState contract
│   │   ├── context.py               # Cached shared artifacts (models + SHAP)
│   │   ├── data_retrieval_agent.py
│   │   ├── risk_assessment_agent.py
│   │   ├── explanation_agent.py
│   │   └── orchestrator.py          # Pipeline driver + CLI
│   └── utils/
│       ├── config.py
│       └── logger.py
├── app.py                  # Streamlit dashboard (streamlit run app.py)
├── .env.example
├── requirements.txt
└── run_pipeline.py         # Full pipeline entry point (7 steps)
```

---

## Week 4: ML Model Evaluation & Fairness Analysis

In Week 4, we trained and evaluated two machine learning models: Logistic Regression and Random Forest for credit risk prediction.

### Model Performance
Both models were evaluated using standard classification metrics such as precision, recall, and AUC-ROC. The models demonstrated strong predictive capability in identifying high-risk borrowers.

### Fairness Analysis
To ensure ethical and unbiased decision-making, fairness metrics were evaluated:

- Demographic Parity Difference: 0.2199 (Bias Detected)
- Equalized Odds Difference: 0.0455 (Acceptable)

The analysis revealed that younger applicants (≤25) had a lower approval rate (~51%) compared to older applicants (>25) with ~73% approval rate. This indicates potential bias in approval distribution across age groups.

### Key Insight
While the model performs well in prediction accuracy, there is a trade-off between performance and fairness. This highlights the importance of fairness-aware model evaluation in financial decision systems.

### Output Artifacts
- Evaluation report: `outputs/reports/evaluation_report.json`
- Trained models stored in: `outputs/models/`

---

## Week 5: SHAP Explainability + Bias Mitigation + Interactive UI

### SHAP Explainability (`src/models/shap_explainer.py`)

- `shap.TreeExplainer` for Random Forest, `shap.LinearExplainer` for Logistic Regression
- **Global plots:** feature-importance bar + beeswarm distribution (per model)
- **Local plots:** waterfall diagrams for correctly-approved, correctly-rejected, and wrongly-approved borrowers
- **Protected-attribute impact plot:** shows how much `age` and `personal_status_sex` individually drive each decision
- Output: `outputs/reports/shap/*.png` + `outputs/reports/shap_report.json`

### Human-Readable Borrower Explanations (`src/models/explain_borrower.py`)

Converts raw SHAP values into plain-English sentences:

```
LOAN DECISION: REJECTED (predicted default)
Default Probability: 90.7%  |  Risk Level: Very High Risk

TOP REASONS THIS APPLICATION WAS FLAGGED:
  1. Checking account status (no checking account) slightly increases default risk.
  2. Loan duration (7 months) slightly increases default risk.
  3. Savings account (no savings account) marginally increases default risk.

FACTORS IN THE APPLICANT'S FAVOUR:
  1. Property / assets (life insurance / savings plan) marginally reduces default risk.
  ...
```

CLI:
```bash
python -m src.models.explain_borrower 100        # explain borrower #100 (RF)
python -m src.models.explain_borrower 100 --lr   # use Logistic Regression
python -m src.models.explain_borrower            # batch: all 250 test borrowers
```

### Bias Mitigation (`src/models/bias_mitigation.py`)

Post-processing two-phase threshold optimisation:
- **Phase 1:** minimise Equalized Odds Difference across groups (TPR + FPR squared loss)
- **Phase 2:** greedy Demographic Parity correction
- Small groups (n < 20) merged with nearest large group to avoid noise
- Group-specific thresholds saved to `outputs/models/fairness_thresholds.json`
- `predict_fair()` is the **only** production prediction method — raw `model.predict()` is no longer used

### Interactive Streamlit Dashboard (`app.py`)

See the [Interactive Dashboard](#interactive-dashboard-apppy) section above. Launch with:

```bash
streamlit run app.py
```

---

---

## Week 6: Multi-Agent Orchestrator (Task 6)

A lightweight, custom agentic framework that coordinates three specialised agents in a sequential pipeline. Each agent has a single responsibility, a standard contract (`run(state) → state`), and participates in a shared `AgentState` that flows through the pipeline.

### Pipeline

```
┌──────────────────────┐     ┌─────────────────────────┐     ┌──────────────────────────┐
│ Data Retrieval Agent │ ──▶ │ Risk Assessment Agent   │ ──▶ │ Explanation Generator    │
│  fetches profile     │     │  predicts P(default)    │     │  SHAP → plain English    │
│  (CSV / Plaid)       │     │  + fairness threshold   │     │  risk + protective lists │
└──────────────────────┘     └─────────────────────────┘     └──────────────────────────┘
                        AgentState flows through all three
```

| Agent | File | Responsibility |
|-------|------|----------------|
| `DataRetrievalAgent` | `src/agents/data_retrieval_agent.py` | Fetches borrower's raw profile. Supports the German Credit test set (default) and has a stub for Plaid-backed retrieval. |
| `RiskAssessmentAgent` | `src/agents/risk_assessment_agent.py` | Runs the trained RF/LR classifier, applies the group-specific fairness threshold from `outputs/models/fairness_thresholds.json`, assigns a risk level. |
| `ExplanationAgent` | `src/agents/explanation_agent.py` | Uses pre-computed SHAP values to build ranked risk/protective factor lists + a plain-English narrative. |

### Orchestrator features

- **Observability** — every agent invocation is captured in `state.agent_trace` with status + elapsed_ms.
- **Error resilience** — failures are caught, recorded with tracebacks, and halt downstream agents (`stop_on_error=True`).
- **Cached context** — models + SHAP values are loaded once per process via `@lru_cache` in `src/agents/context.py`, so subsequent invocations are ~1 ms instead of ~200 ms.
- **No external dependencies** — pure-Python, no LangChain / LangGraph, fully auditable for a course project.

### CLI

```bash
# Run the full 3-agent pipeline for one borrower (RF model)
python -m src.agents.orchestrator 100

# Use Logistic Regression instead
python -m src.agents.orchestrator 100 --lr

# Save the full AgentState (profile + decision + narrative + trace) to JSON
python -m src.agents.orchestrator 100 --save outputs/reports/decision_100.json
```

Example output:

```
 Agent Execution Trace:
 Agent                       Status     Time (ms)
 data_retrieval_agent        success        183.9
 risk_assessment_agent       success          0.5
 explanation_agent           success          0.4

 Final Decision:
   Borrower ID          : #100
   Default probability  : 90.7%
   Risk level           : Very High Risk
   Decision             : REJECTED
   Fairness threshold   : 0.510  (personal_status_sex=male : single)
   Ground truth         : Default
```

### Programmatic usage

```python
from src.agents import Orchestrator

orch  = Orchestrator()
state = orch.run(borrower_id=100, model="rf")

print(state.decision)              # "REJECTED"
print(state.default_probability)   # 0.9068
print(state.narrative)             # full English explanation
print(state.agent_trace)           # step-by-step timing
```

### Dashboard integration

The Streamlit dashboard (`app.py`) now drives its decisions through the orchestrator. A new **Agent Pipeline Trace** panel shows the three cards (Data Retrieval / Risk Assessment / Explanation Generator), each with its own status badge + execution time, plus the fairness threshold applied for that borrower.

---

## Validation Suite

Four independent validation scripts live under `src/validation/`. Each one
prints a human-readable summary, writes a JSON report to
`outputs/reports/validation/`, and exits with a non-zero status when checks
fail (so they can be wired into CI).

| Script | Purpose | Pass rule |
|---|---|---|
| `cross_validation.py` | K-fold CV on Lending Club (LR + RF) | CV(AUC) = std/mean < 0.05 |
| `fairness_validation.py` | DPD + EOD on gender, age, foreign_worker (German Credit) | DPD ≤ 0.10 AND EOD ≤ 0.10 |
| `explainability_validation.py` | SHAP top-5 features match domain expectations | ≥ 3 of top-5 in expected set |
| `pipeline_validation.py` | End-to-end orchestrator smoke tests + determinism check | All per-borrower + global checks pass |

### Run individually
```bash
python -m src.validation.cross_validation --k 5
python -m src.validation.fairness_validation --threshold 0.10
python -m src.validation.explainability_validation --min-overlap 3
python -m src.validation.pipeline_validation --ids 1 25 50 100 150 200
```

### Run everything
```bash
python -m src.validation.run_all
python -m src.validation.run_all --skip cross      # skip slow CV
```

Outputs land in:
```
outputs/reports/validation/
├── cross_validation.json
├── fairness_validation.json
├── explainability_validation.json
├── pipeline_validation.json
└── combined_validation_report.json
```

## Upcoming

- **Task 7** — Full benchmark evaluation report and final presentation
