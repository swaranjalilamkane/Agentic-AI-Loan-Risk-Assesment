"""
Loan Risk Assessment – Borrower Explanation Dashboard
Run with:  streamlit run app.py
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── path fix so src/ is importable ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

# ── auto-load .env so Plaid creds reach os.getenv() without shell sourcing ──
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="SAAi – Agentic AI Loan Risk Assessment",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ---- native header: keep visible, style it, inject title via ::after ---- */
[data-testid="stHeader"] {
    background: #1a2332 !important;
    border-bottom: 1px solid #2e3f55;
}

/* Title text injected into the header bar */
[data-testid="stHeader"]::after {
    content: "SAAi-  Agentic AI Loan Risk Assessment";
    position: absolute;
    left: 50%;
    transform: translateX(-50%);
    top: 50%;
    margin-top: -10px;
    font-size: 1.50rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: 0.02em;
    white-space: nowrap;
    pointer-events: none;
}

/* Hide only the deploy button — KEEP the toolbar itself because the
   sidebar expand button also lives inside it. */
[data-testid="stAppDeployButton"] { display: none !important; }
[data-testid="stMainMenu"] { display: none !important; }
#MainMenu { display: none !important; }
footer { display: none !important; }

/* ---- global ---- */
[data-testid="stAppViewContainer"] {
    background: #f0f4f8;
}

/* ---- sidebar ---- */
[data-testid="stSidebar"] {
    background: #1a2332;
}
/* Remove Streamlit's default top padding so content starts flush at top */
[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
}
[data-testid="stSidebar"] * {
    color: #e8edf3 !important;
}
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stRadio label {
    color: #b0bec5 !important;
}

/* ---- sidebar collapse button (arrow inside sidebar when expanded) ---- */
[data-testid="stSidebarCollapseButton"] button {
    background: #2e3f55 !important;
    border: 1px solid #3d5166 !important;
    color: #ffffff !important;
}
[data-testid="stSidebarCollapseButton"] button svg {
    fill: #ffffff !important;
    color: #ffffff !important;
}
[data-testid="stSidebarCollapseButton"] button:hover {
    background: #1565c0 !important;
}

/* ---- sidebar EXPAND button (shows in header when sidebar is collapsed) ---- */
/* The button element itself carries data-testid="stExpandSidebarButton" */
[data-testid="stExpandSidebarButton"] {
    display: inline-flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    background: #1565c0 !important;
    border: 2px solid #ffffff !important;
    color: #ffffff !important;
    border-radius: 50% !important;
    width: 38px !important;
    height: 38px !important;
    min-width: 38px !important;
    padding: 0 !important;
    margin: 0 8px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.35) !important;
    align-items: center !important;
    justify-content: center !important;
}
[data-testid="stExpandSidebarButton"] svg,
[data-testid="stExpandSidebarButton"] * {
    color: #ffffff !important;
    fill: #ffffff !important;
}
[data-testid="stExpandSidebarButton"]:hover {
    background: #1976d2 !important;
    transform: scale(1.08);
}

/* ---- metric cards ---- */
.metric-card {
    background: white;
    border-radius: 14px;
    padding: 20px 24px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    text-align: center;
}
.metric-title {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #78909c;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    line-height: 1.1;
}

/* ---- decision badge ---- */
.badge-rejected {
    background: #fdecea;
    border: 2px solid #e53935;
    color: #b71c1c;
    border-radius: 50px;
    padding: 10px 28px;
    font-size: 1.05rem;
    font-weight: 700;
    display: inline-block;
    letter-spacing: 0.04em;
}
.badge-approved {
    background: #e8f5e9;
    border: 2px solid #43a047;
    color: #1b5e20;
    border-radius: 50px;
    padding: 10px 28px;
    font-size: 1.05rem;
    font-weight: 700;
    display: inline-block;
    letter-spacing: 0.04em;
}

/* ---- factor rows ---- */
.factor-row {
    display: flex;
    align-items: center;
    gap: 12px;
    background: white;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
    box-shadow: 0 1px 5px rgba(0,0,0,0.05);
}
.factor-icon { font-size: 1.2rem; min-width: 28px; }
.factor-text { flex: 1; }
.factor-label { font-size: 0.78rem; color: #78909c; font-weight: 600; text-transform: uppercase; }
.factor-sentence { font-size: 0.93rem; color: #263238; margin-top: 2px; }
.factor-bar-track {
    width: 110px; height: 8px; background: #eceff1;
    border-radius: 4px; overflow: hidden; flex-shrink: 0;
}
.factor-bar-fill-risk { height: 100%; background: #ef5350; border-radius: 4px; }
.factor-bar-fill-safe { height: 100%; background: #66bb6a; border-radius: 4px; }
.factor-shap {
    font-size: 0.8rem; font-weight: 700; min-width: 50px; text-align: right;
}

/* ---- section header ---- */
.section-header {
    font-size: 0.85rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #546e7a;
    margin: 20px 0 10px 0;
    padding-left: 4px;
    border-left: 3px solid #1565c0;
    padding-left: 10px;
}

/* ---- borrower raw table ---- */
.raw-table-wrapper {
    background: white;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 1px 8px rgba(0,0,0,0.06);
}
</style>
""",
    unsafe_allow_html=True,
)

# ── cache the heavy artifacts (loaded once per session) ─────────────────────
@st.cache_resource(show_spinner="Loading models and computing SHAP values…")
def load_artifacts():
    from src.models.explain_borrower import _load_pipeline_artifacts
    return _load_pipeline_artifacts()


# ── helper: gauge chart ──────────────────────────────────────────────────────
def _gauge(prob: float, risk_level: str) -> go.Figure:
    color = (
        "#ef5350" if prob >= 0.55
        else "#ffa726" if prob >= 0.40
        else "#66bb6a"
    )
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(prob * 100, 1),
            number={"suffix": "%", "font": {"size": 36, "color": color}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#90a4ae",
                    "tickfont": {"size": 10},
                },
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 25],  "color": "#e8f5e9"},
                    {"range": [25, 40], "color": "#f9fbe7"},
                    {"range": [40, 55], "color": "#fff8e1"},
                    {"range": [55, 75], "color": "#fce4ec"},
                    {"range": [75, 100],"color": "#ffebee"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 3},
                    "thickness": 0.8,
                    "value": prob * 100,
                },
            },
            title={
                "text": f"<b>{risk_level}</b>",
                "font": {"size": 14, "color": color},
            },
        )
    )
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="white",
        font={"family": "Inter, sans-serif"},
    )
    return fig


# ── helper: factor bar (HTML) ────────────────────────────────────────────────
def _factor_html(factors: list[dict], kind: str) -> str:
    if not factors:
        return "<p style='color:#90a4ae;font-size:0.9rem;'>None found.</p>"

    max_abs = max(abs(f["shap"]) for f in factors) or 1
    rows = []
    for f in factors:
        pct = int(abs(f["shap"]) / max_abs * 100)
        icon = "🔴" if kind == "risk" else "🟢"
        bar_class = "factor-bar-fill-risk" if kind == "risk" else "factor-bar-fill-safe"
        shap_color = "#ef5350" if kind == "risk" else "#43a047"
        sign = "+" if f["shap"] > 0 else ""
        rows.append(
            f"""
            <div class="factor-row">
              <div class="factor-icon">{icon}</div>
              <div class="factor-text">
                <div class="factor-label">{f['factor']}</div>
                <div class="factor-sentence">{f['value']}</div>
              </div>
              <div class="factor-bar-track">
                <div class="{bar_class}" style="width:{pct}%"></div>
              </div>
              <div class="factor-shap" style="color:{shap_color}">
                {sign}{f['shap']:.3f}
              </div>
            </div>
            """
        )
    return "".join(rows)


# ── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:10px;
                    padding:14px 4px 14px 4px;
                    border-bottom:1px solid #2e3f55;margin-bottom:16px;">
            <span style="font-size:1.8rem;line-height:1;">🏦</span>
            <span style="font-size:0.95rem;font-weight:700;color:#ffffff;
                         letter-spacing:0.02em;line-height:1.1;">
                SAAi Loan Risk
            </span>
        </div>
        <div style="padding:0 0 4px 0;font-size:0.65rem;font-weight:700;
                    text-transform:uppercase;letter-spacing:0.1em;color:#546e7a;">
            Data Source
        </div>
        """,
        unsafe_allow_html=True,
    )
    data_source = st.radio(
        "Choose data source",
        options=["German Credit (test set)", "Plaid Live (sandbox)"],
        index=0,
        label_visibility="collapsed",
        help=(
            "German Credit = cached 250-row test set, ground truth known.\n"
            "Plaid Live    = pull a fresh borrower's bank data from the "
            "Plaid sandbox and score it against the trained model."
        ),
    )
    is_plaid = data_source.startswith("Plaid")

    st.markdown(
        """
        <div style="padding:14px 0 4px 0;font-size:0.65rem;font-weight:700;
                    text-transform:uppercase;letter-spacing:0.1em;color:#546e7a;">
            Borrower Lookup
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── CSV mode controls ───────────────────────────────────────────────────
    borrower_idx = 0
    application_form: dict = {}
    institution_choice = "ins_109508"
    fetch_clicked = False

    if not is_plaid:
        borrower_idx = st.number_input(
            "Borrower Index (test set)",
            min_value=0,
            max_value=249,
            value=0,
            step=1,
            help="The German Credit test set has 250 borrowers (index 0–249).",
        )
    else:
        # ── Plaid mode controls: application form + institution rotation ──
        st.caption("Application form (sent with Plaid bank data)")
        application_form = {
            "amount":   st.number_input("Loan amount ($)",
                                         min_value=500, max_value=50000,
                                         value=5000, step=500),
            "duration": st.number_input("Duration (months)",
                                         min_value=6, max_value=72,
                                         value=24, step=6),
            "age":      st.number_input("Age", min_value=18, max_value=90,
                                         value=35, step=1),
            "purpose":  st.selectbox(
                "Purpose",
                options=[
                    "car (new)", "car (used)", "furniture/equipment",
                    "radio/television", "domestic appliances", "repairs",
                    "education", "vacation", "retraining", "business",
                    "other purpose",
                ],
                index=10,
            ),
            "personal_status_sex": st.selectbox(
                "Personal status / sex",
                options=[
                    "male : single",
                    "male : divorced/separated",
                    "male : married/widowed",
                    "female : divorced/separated/married",
                ],
                index=0,
            ),
        }

        institution_choice = st.selectbox(
            "Sandbox institution",
            options=["auto-rotate",
                     "ins_109508 (First Platypus)",
                     "ins_3 (Chase)",
                     "ins_4 (Wells Fargo)",
                     "ins_5 (Bank of America)",
                     "ins_6 (Citi)"],
            index=0,
            help=(
                "auto-rotate cycles through sandbox institutions each click "
                "so every fetch is a new synthetic borrower."
            ),
        )

        # The actual button that triggers a fetch
        fetch_clicked = st.button(
            "🔄  Fetch new Plaid record",
            width="stretch",
            type="primary",
        )

    model_choice = st.radio(
        "Model",
        options=["Random Forest", "Logistic Regression"],
        index=0,
    )
    model_key = "rf" if model_choice == "Random Forest" else "lr"

    top_n = st.slider("Factors to display", min_value=3, max_value=10, value=5)

    st.markdown(
        """
        <hr style="border-color:#2e3f55;margin:20px 0 12px 0"/>
        <div style="font-size:0.72rem;color:#607d8b;line-height:1.6;">
        <b style="color:#90caf9;">SHAP values</b> measure each feature's
        contribution to the model's decision.<br><br>
        🔴 Positive → pushes toward <b>default</b><br>
        🟢 Negative → pushes toward <b>approval</b>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── main panel ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <p style="color:#78909c;font-size:0.88rem;margin-bottom:20px;margin-top:-4px;">
      AI-powered credit decision with SHAP explainability &nbsp;·&nbsp;
      German Credit Dataset &nbsp;·&nbsp; Fairness-aware predictions
    </p>
    """,
    unsafe_allow_html=True,
)

# Load models
try:
    artifacts = load_artifacts()
except Exception as e:
    st.error(
        f"Could not load models. Make sure you have run the pipeline first.\n\n{e}"
    )
    st.stop()

m          = artifacts[model_key]
fn         = artifacts["feature_names"]
raw_test   = artifacts["raw_test"]
y_test     = artifacts["y_test"]
n_borrowers = len(raw_test)

# ── orchestrator run — branch on data source ────────────────────────────────
from src.agents import Orchestrator
from src.agents.data_retrieval_agent import DataRetrievalAgent
from src.agents.risk_assessment_agent import RiskAssessmentAgent
from src.agents.explanation_agent import ExplanationAgent

# Sandbox institutions to rotate through in "auto-rotate" mode
_PLAID_INSTITUTIONS = ["ins_109508", "ins_3", "ins_4", "ins_5", "ins_6"]

if not is_plaid:
    # ─── CSV path (unchanged) ─────────────────────────────────────────────
    if borrower_idx >= n_borrowers:
        st.warning(
            f"Index {borrower_idx} out of range. "
            f"Test set has {n_borrowers} borrowers."
        )
        st.stop()

    orchestrator = Orchestrator()
    agent_state  = orchestrator.run(
        borrower_id   = borrower_idx,
        model         = model_key,
        top_n_factors = top_n,
    )

else:
    # ─── Plaid Live path ──────────────────────────────────────────────────
    if not (os.getenv("PLAID_CLIENT_ID") and os.getenv("PLAID_SECRET")):
        st.error(
            "🔑 PLAID_CLIENT_ID / PLAID_SECRET are not set.\n\n"
            "Add them to your `.env` file (or shell export them) and reload "
            "the page. Dashboard → Team Settings → Keys."
        )
        st.stop()

    # Click counter used for auto-rotate AND cache-busting
    if "plaid_click_count" not in st.session_state:
        st.session_state.plaid_click_count = 0
    if fetch_clicked:
        st.session_state.plaid_click_count += 1
        st.session_state.pop("plaid_agent_state", None)  # force refetch

    # Resolve institution id (strip the "(Name)" suffix)
    if institution_choice.startswith("auto-rotate"):
        inst_id = _PLAID_INSTITUTIONS[
            st.session_state.plaid_click_count % len(_PLAID_INSTITUTIONS)
        ]
    else:
        inst_id = institution_choice.split()[0]   # e.g. "ins_109508"

    # Need a cached state — run the pipeline if no cache or user asked for it
    needs_run = (
        fetch_clicked
        or "plaid_agent_state" not in st.session_state
        or st.session_state.get("plaid_model") != model_key
        or st.session_state.get("plaid_top_n") != top_n
        or st.session_state.get("plaid_form") != application_form
        or st.session_state.get("plaid_inst") != inst_id
    )

    if needs_run:
        if "plaid_agent_state" not in st.session_state:
            st.info(
                "Click **🔄 Fetch new Plaid record** in the sidebar to pull "
                "a fresh borrower from the Plaid sandbox and score them."
            )
        with st.spinner(f"Calling Plaid sandbox ({inst_id}) and running agents…"):
            orchestrator = Orchestrator(agents=[
                DataRetrievalAgent(
                    source         = "plaid",
                    application    = application_form,
                    institution_id = inst_id,
                ),
                RiskAssessmentAgent(),
                ExplanationAgent(),
            ])
            agent_state = orchestrator.run(
                borrower_id   = 0,      # unused for Plaid but required by state
                model         = model_key,
                top_n_factors = top_n,
            )
        st.session_state.plaid_agent_state = agent_state
        st.session_state.plaid_model       = model_key
        st.session_state.plaid_top_n       = top_n
        st.session_state.plaid_form        = dict(application_form)
        st.session_state.plaid_inst        = inst_id
    else:
        agent_state = st.session_state.plaid_agent_state

if agent_state.errors:
    for err in agent_state.errors:
        st.error(f"Agent error: {err}")
    st.stop()

# Re-build the per-factor dict the rest of the UI expects
result = {
    "probability":        agent_state.default_probability,
    "risk_level":         agent_state.risk_level,
    "decision":           agent_state.decision + (
        " (predicted default)" if agent_state.decision == "REJECTED"
        else " (predicted good credit)"
    ),
    "actual_label":       agent_state.actual_label,
    "risk_factors":       agent_state.risk_factors,
    "protective_factors": agent_state.protective_factors,
    "factors_detail":     agent_state.shap_report,
    "narrative":          agent_state.narrative,
}

prob        = result["probability"]
risk_level  = result["risk_level"]
decision    = result["decision"]
is_rejected = decision.startswith("REJECTED")
actual      = result["actual_label"]

# ── row 1: identity + gauge + key metrics ────────────────────────────────────
col_id, col_gauge, col_metrics = st.columns([1.6, 1.8, 2.6])

with col_id:
    if not is_plaid:
        id_label = f"#{borrower_idx}"
        id_sub = (
            f"<div style='font-size:0.8rem;color:#78909c;margin-top:6px;'>"
            f"Ground truth: <b style=\"color:{'#c62828' if actual==1 else '#2e7d32'};\">"
            f"{'Default' if actual==1 else 'Good Credit'}</b></div>"
        )
    else:
        id_label = f"LIVE #{st.session_state.plaid_click_count}"
        id_sub = (
            f"<div style='font-size:0.8rem;color:#78909c;margin-top:6px;'>"
            f"Institution: <b style='color:#455a64;'>"
            f"{st.session_state.get('plaid_inst', '—')}</b></div>"
            f"<div style='font-size:0.75rem;color:#90a4ae;margin-top:4px;'>"
            f"Source: <b style='color:#1976d2;'>Plaid Sandbox</b></div>"
        )
    st.markdown(
        f"""
        <div class="metric-card" style="height:100%;min-height:220px;">
          <div class="metric-title">Borrower</div>
          <div class="metric-value" style="color:#1565c0;">{id_label}</div>
          <div style="margin-top:14px;">
            {'<span class="badge-rejected">⚠ REJECTED</span>'
             if is_rejected
             else '<span class="badge-approved">✓ APPROVED</span>'}
          </div>
          <div style="margin-top:14px;font-size:0.8rem;color:#78909c;">
            Model: <b style="color:#455a64;">{model_choice}</b>
          </div>
          {id_sub}
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_gauge:
    st.plotly_chart(
        _gauge(prob, risk_level),
        width="stretch",
        config={"displayModeBar": False},
    )

with col_metrics:
    # 4 mini metric cards
    risk_factors      = result["risk_factors"]
    protective_factors = result["protective_factors"]
    top_risk   = risk_factors[0]["factor"]   if risk_factors   else "—"
    top_safe   = protective_factors[0]["factor"] if protective_factors else "—"

    if not is_plaid:
        # Compare model prediction against historical ground truth
        pred_correct = (is_rejected and actual == 1) or (not is_rejected and actual == 0)
        if pred_correct:
            correct_label    = "✓ Correct"
            correct_subtitle = ("True Positive" if is_rejected else "True Negative")
        else:
            correct_label    = "✗ Incorrect"
            # False Positive: rejected someone who would have paid back
            # False Negative: approved someone who actually defaulted
            correct_subtitle = ("False Positive" if is_rejected else "False Negative")
        correct_color = "#2e7d32" if pred_correct else "#c62828"
        correct_title = "Prediction vs Ground Truth"
    else:
        # No ground truth for live Plaid — show the decision threshold used
        thr   = agent_state.fairness_threshold_used
        grp   = agent_state.group_for_threshold or "default"
        correct_label    = f"{thr:.2f}" if thr is not None else "0.50"
        correct_subtitle = f"Fairness threshold ({grp})"
        correct_color    = "#1565c0"
        correct_title    = "Decision Threshold"

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card" style="margin-bottom:12px;">
              <div class="metric-title">Default Probability</div>
              <div class="metric-value" style="color:{'#ef5350' if prob>=0.55 else '#ffa726' if prob>=0.40 else '#43a047'};">
                {prob:.1%}
              </div>
            </div>
            <div class="metric-card">
              <div class="metric-title">Top Risk Factor</div>
              <div style="font-size:0.95rem;font-weight:600;color:#37474f;margin-top:4px;">
                {top_risk}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card" style="margin-bottom:12px;">
              <div class="metric-title">{correct_title}</div>
              <div class="metric-value" style="font-size:1.05rem;color:{correct_color};">
                {correct_label}
              </div>
              <div style="font-size:0.72rem;color:#78909c;margin-top:2px;
                          font-weight:500;letter-spacing:0.02em;">
                {correct_subtitle}
              </div>
            </div>
            <div class="metric-card">
              <div class="metric-title">Top Protective Factor</div>
              <div style="font-size:0.95rem;font-weight:600;color:#37474f;margin-top:4px;">
                {top_safe}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ── row 2: risk factors + protective factors ──────────────────────────────────
col_risk, col_prot = st.columns(2)

with col_risk:
    st.markdown(
        '<div class="section-header" style="border-color:#ef5350;"> &nbsp;Top Risk Factors</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        _factor_html(result["risk_factors"], "risk"),
        unsafe_allow_html=True,
    )

with col_prot:
    st.markdown(
        '<div class="section-header" style="border-color:#43a047;">&nbsp;Protective Factors</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        _factor_html(result["protective_factors"], "safe"),
        unsafe_allow_html=True,
    )

# ── row 3: SHAP waterfall (if plot exists) + raw profile ─────────────────────
st.markdown("<br>", unsafe_allow_html=True)
col_wf, col_raw = st.columns([1.4, 1])

with col_wf:
    st.markdown(
        '<div class="section-header">📊 &nbsp;Full SHAP Breakdown (all features)</div>',
        unsafe_allow_html=True,
    )

    all_factors = sorted(result["factors_detail"], key=lambda x: x["shap"])
    feat_labels = [f["factor"] for f in all_factors]
    shap_vals   = [f["shap"]  for f in all_factors]
    colors      = ["#ef5350" if v > 0 else "#66bb6a" for v in shap_vals]

    fig_bar = go.Figure(
        go.Bar(
            x=shap_vals,
            y=feat_labels,
            orientation="h",
            marker_color=colors,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "SHAP: %{x:.4f}<br>"
                "<extra></extra>"
            ),
        )
    )
    fig_bar.add_vline(x=0, line_width=1.5, line_color="#546e7a")
    fig_bar.update_layout(
        height=max(380, len(feat_labels) * 22),
        margin=dict(l=10, r=20, t=10, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(
            title=dict(
                text="SHAP value  (red = risk ↑  |  green = risk ↓)",
                font=dict(size=12, color="#37474f"),
            ),
            tickfont=dict(size=11, color="#37474f"),
            gridcolor="#eceff1",
            zeroline=False,
        ),
        yaxis=dict(
            automargin=True,
            tickfont=dict(size=12, color="#263238"),
        ),
        bargap=0.25,
        font=dict(family="Inter, sans-serif", color="#263238"),
    )
    st.plotly_chart(fig_bar, width="stretch", config={"displayModeBar": False})

with col_raw:
    if not is_plaid:
        st.markdown(
            '<div class="section-header">📋 &nbsp;Borrower Profile</div>',
            unsafe_allow_html=True,
        )
        raw_row = raw_test.iloc[borrower_idx]

        # Select informative columns only (drop internal ones)
        display_cols = [
            "age", "duration", "amount", "credit_history", "purpose",
            "status", "savings", "employment_duration", "installment_rate",
            "housing", "job", "property", "other_debtors",
            "present_residence", "number_credits", "people_liable",
            "personal_status_sex", "foreign_worker",
            "credit_amount_per_duration", "credit_per_person",
        ]
        display_cols = [c for c in display_cols if c in raw_row.index]

        profile_df = pd.DataFrame(
            {"Feature": display_cols,
             "Value":   [str(raw_row[c]) for c in display_cols]}
        )
        st.dataframe(
            profile_df.set_index("Feature"),
            width="stretch",
            height=min(600, len(display_cols) * 36 + 40),
        )
    else:
        # ─── Plaid live signals + provenance ───────────────────────────────
        st.markdown(
            '<div class="section-header">📡 &nbsp;Plaid Live Signals</div>',
            unsafe_allow_html=True,
        )
        live = agent_state.raw_profile.get("_plaid_live_signals", {}) or {}
        if live:
            live_df = pd.DataFrame(
                {"Signal": list(live.keys()),
                 "Value":  [str(v) for v in live.values()]}
            )
            st.dataframe(
                live_df.set_index("Signal"),
                width="stretch",
                height=min(320, len(live) * 36 + 40),
            )
        else:
            st.caption("No live signals captured.")

        st.markdown(
            '<div class="section-header" style="margin-top:18px;">'
            '🧾 &nbsp;Feature Provenance</div>',
            unsafe_allow_html=True,
        )
        prov = agent_state.feature_provenance or {}
        if prov:
            # Bucket features by their source
            groups: dict[str, list[str]] = {}
            for feat, src in prov.items():
                groups.setdefault(src, []).append(feat)

            rows = []
            for src_name in ("plaid", "application", "engineered", "default"):
                if src_name not in groups:
                    continue
                rows.append({
                    "Source": src_name,
                    "Count":  len(groups[src_name]),
                    "Features": ", ".join(sorted(groups[src_name])),
                })
            prov_df = pd.DataFrame(rows)
            st.dataframe(
                prov_df.set_index("Source"),
                width="stretch",
                height=min(220, len(rows) * 90 + 40),
            )
        else:
            st.caption("No provenance metadata available.")

        defs_used = agent_state.raw_profile.get("_plaid_defaults_used", []) or []
        if defs_used:
            st.caption(
                f"ℹ️ Application / default values used for: "
                f"**{', '.join(defs_used)}**"
            )

# ── footer ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <hr style="border-color:#cfd8dc;margin:32px 0 12px 0"/>
    <div style="text-align:center;font-size:0.75rem;color:#90a4ae;">
      SAAi – Agentic AI Loan Risk Assessment &nbsp;|&nbsp;
      German Credit (1000) · Lending Club (CV) · Plaid Sandbox (live) &nbsp;|&nbsp;
      SHAP Explainability &nbsp;|&nbsp;
      Fairness-aware predictions
    </div>
    """,
    unsafe_allow_html=True,
)
