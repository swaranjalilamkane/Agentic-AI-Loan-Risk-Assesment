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

# ── page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Loan Risk Assessment",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ---- global ---- */
[data-testid="stAppViewContainer"] {
    background: #f0f4f8;
}
[data-testid="stSidebar"] {
    background: #1a2332;
}
[data-testid="stSidebar"] * {
    color: #e8edf3 !important;
}
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stRadio label {
    color: #b0bec5 !important;
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
        <div style="text-align:center;padding:20px 0 10px 0;">
          <span style="font-size:2.5rem;">🏦</span><br>
          <span style="font-size:1.1rem;font-weight:700;color:#90caf9;">
            Loan Risk AI
          </span><br>
          <span style="font-size:0.75rem;color:#607d8b;">
            SHAP Explainability Dashboard
          </span>
        </div>
        <hr style="border-color:#2e3f55;margin:10px 0 20px 0"/>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Borrower Lookup")
    borrower_idx = st.number_input(
        "Borrower Index (test set)",
        min_value=0,
        max_value=249,
        value=0,
        step=1,
        help="The German Credit test set has 250 borrowers (index 0–249).",
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
    <h1 style="font-size:1.8rem;font-weight:800;color:#1a2332;margin-bottom:4px;">
      Loan Risk Assessment
    </h1>
    <p style="color:#78909c;font-size:0.95rem;margin-bottom:24px;">
      AI-powered credit decision with SHAP explainability
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

if borrower_idx >= n_borrowers:
    st.warning(f"Index {borrower_idx} out of range. Test set has {n_borrowers} borrowers.")
    st.stop()

# Build explanation on the fly
from src.models.explain_borrower import explain_borrower

result = explain_borrower(
    shap_values     = m["shap_vals"][borrower_idx],
    raw_row         = raw_test.iloc[borrower_idx],
    feature_names   = fn,
    base_value      = m["base_value"],
    predicted_prob  = m["probs"][borrower_idx],
    predicted_label = int(m["preds"][borrower_idx]),
    actual_label    = int(y_test.iloc[borrower_idx]),
    borrower_id     = borrower_idx,
    top_n           = top_n,
)

prob        = result["probability"]
risk_level  = result["risk_level"]
decision    = result["decision"]
is_rejected = decision.startswith("REJECTED")
actual      = result["actual_label"]

# ── row 1: identity + gauge + key metrics ────────────────────────────────────
col_id, col_gauge, col_metrics = st.columns([1.6, 1.8, 2.6])

with col_id:
    st.markdown(
        f"""
        <div class="metric-card" style="height:100%;min-height:220px;">
          <div class="metric-title">Borrower</div>
          <div class="metric-value" style="color:#1565c0;">#{borrower_idx}</div>
          <div style="margin-top:14px;">
            {'<span class="badge-rejected">⚠ REJECTED</span>'
             if is_rejected
             else '<span class="badge-approved">✓ APPROVED</span>'}
          </div>
          <div style="margin-top:14px;font-size:0.8rem;color:#78909c;">
            Model: <b style="color:#455a64;">{model_choice}</b>
          </div>
          <div style="font-size:0.8rem;color:#78909c;margin-top:6px;">
            Ground truth:
            <b style="color:{'#c62828' if actual==1 else '#2e7d32'};">
              {'Default' if actual==1 else 'Good Credit'}
            </b>
          </div>
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
    correct    = (
        ("✓ Correct" if (is_rejected and actual == 1) or (not is_rejected and actual == 0)
         else "✗ Wrong")
    )
    correct_color = "#2e7d32" if correct.startswith("✓") else "#c62828"

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
              <div class="metric-title">Prediction</div>
              <div class="metric-value" style="font-size:1.1rem;color:{correct_color};">
                {correct}
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
            title="SHAP value  (red = risk ↑  |  green = risk ↓)",
            title_font_size=11,
            gridcolor="#eceff1",
        ),
        yaxis=dict(automargin=True, tickfont=dict(size=11)),
        bargap=0.25,
        font={"family": "Inter, sans-serif"},
    )
    st.plotly_chart(fig_bar, width="stretch", config={"displayModeBar": False})

with col_raw:
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
        {"Feature": display_cols, "Value": [str(raw_row[c]) for c in display_cols]}
    )
    st.dataframe(
        profile_df.set_index("Feature"),
        width="stretch",
        height=min(600, len(display_cols) * 36 + 40),
    )

# ── footer ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <hr style="border-color:#cfd8dc;margin:32px 0 12px 0"/>
    <div style="text-align:center;font-size:0.75rem;color:#90a4ae;">
      SAAi – Agentic AI Loan Risk Assessment &nbsp;|&nbsp;
      German Credit Dataset (1000 samples) &nbsp;|&nbsp;
      SHAP Explainability &nbsp;|&nbsp;
      Fairness-aware predictions
    </div>
    """,
    unsafe_allow_html=True,
)
