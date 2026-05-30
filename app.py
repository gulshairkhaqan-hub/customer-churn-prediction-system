# app.py
# Streamlit dashboard for the Customer Churn Prediction system.
#
# This app is a READER, not a trainer:
#   - It loads the pre-saved outputs from outputs/  (CSVs and PNGs).
#   - It loads the pre-saved models from   models/  (joblib).
#   - It does NOT retrain anything.
#
# If a required artifact is missing, the affected section degrades
# gracefully with a helpful st.error / st.warning instead of crashing.

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

SCORING_CSV       = OUTPUTS_DIR / "weekly_scoring_week_1.csv"
RECOMMENDATIONS_CSV = OUTPUTS_DIR / "retention_recommendations.csv"
RAW_DATA_CSV      = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Plot files
PLOT_CHURN_DIST          = OUTPUTS_DIR / "churn_distribution.png"
PLOT_RISK_SEGMENTATION   = OUTPUTS_DIR / "risk_segmentation.png"
PLOT_MODEL_COMPARISON    = OUTPUTS_DIR / "model_comparison.png"
PLOT_SHAP_SUMMARY        = OUTPUTS_DIR / "shap_summary.png"
PLOT_CORRELATION_HEATMAP = OUTPUTS_DIR / "correlation_heatmap.png"
PLOT_CHURN_PROB_DIST     = OUTPUTS_DIR / "churn_prob_distribution.png"

# Model files
MODEL_PATHS = {
    "Logistic Regression": MODELS_DIR / "logistic_regression.pkl",
    "XGBoost":             MODELS_DIR / "xgboost_model.pkl",
    "LightGBM":            MODELS_DIR / "lightgbm_model.pkl",
}
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"


# ----------------------------------------------------------------------
# Page config + dark theme CSS
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
    /* Background */
    .stApp {
        background-color: #0f172a;
        color: #e2e8f0;
    }
    /* Sidebar is collapsed but we still tame its hidden surface so it
       can't leak white background if the user pops it open. */
    [data-testid="stSidebar"] {
        background-color: #1e293b;
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0;
    }

    /* Give the main content enough top padding so the navbar row is
       never clipped by the Streamlit header bar. */
    .block-container {
        padding-top: 3.5rem !important;
    }

    /* Headings */
    h1, h2, h3, h4 {
        color: #f1f5f9 !important;
    }

    /* ----------------------------------------------------------------
       Top navigation bar
       ---------------------------------------------------------------- */
    /* The navbar is a row of st.button widgets laid out with st.columns,
       which already renders as an equal-width flex row. We style the
       buttons by their Streamlit "kind":
         - secondary = inactive tab
         - primary   = active tab
       Colors are forced with !important and applied to the nested label
       node (button *) so the text stays visible across Streamlit versions
       and on Streamlit Cloud. */

    /* Keep the nav row on a single line, centered, with a small gap. */
    div[data-testid="stHorizontalBlock"]:first-of-type {
        display: flex !important;
        justify-content: center !important;
        flex-wrap: nowrap !important;
        gap: 8px !important;
        padding: 18px 0 12px 0 !important;
        border-bottom: 1px solid #334155 !important;
        margin-bottom: 24px !important;
    }
    /* Make sure each nav column doesn't clip its button vertically. */
    div[data-testid="stHorizontalBlock"]:first-of-type [data-testid="column"] {
        display: flex !important;
        align-items: stretch !important;
    }

    /* Inactive tabs. */
    .stButton button,
    .stButton button[kind="secondary"] {
        background-color: #1e293b !important;
        color: #94a3b8 !important;
        border: 1px solid #475569 !important;
        border-radius: 8px;
        font-weight: 700 !important;
        min-height: 46px;
        width: 100%;
        white-space: nowrap;
        transition: all 0.25s ease !important;
    }
    .stButton button[kind="secondary"] * {
        color: #94a3b8 !important;
        font-weight: 700 !important;
    }

    /* Hover state. */
    .stButton button:hover {
        background-color: #6366f1 !important;
        border-color: #818cf8 !important;
        color: #ffffff !important;
        transform: translateY(-4px) !important;
        box-shadow: 0 8px 20px rgba(99, 102, 241, 0.5) !important;
        cursor: pointer !important;
    }
    .stButton button:hover * {
        color: #ffffff !important;
    }
    .stButton button:active {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4) !important;
    }

    /* Active tab (primary type): solid indigo, white text,
       3px green bottom border. */
    .stButton button[kind="primary"] {
        background-color: #6366f1 !important;
        color: #ffffff !important;
        border: 1px solid #6366f1 !important;
        border-bottom: 3px solid #22c55e !important;
    }
    .stButton button[kind="primary"] * {
        color: #ffffff !important;
    }

    /* ----------------------------------------------------------------
       Metric cards (uniform dark navy with indigo accent + hover lift)
       ---------------------------------------------------------------- */
    [data-testid="stMetric"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-left: 4px solid #6366f1;
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.25);
        transition: all 0.3s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-6px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4);
        border-color: #6366f1;
        cursor: default;
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-weight: 600;
    }
    [data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
        font-size: 1.9rem !important;
    }

    /* Risk pills */
    .risk-pill {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.5px;
    }
    .risk-pill.high   { background:#7f1d1d; color:#fee2e2; }
    .risk-pill.medium { background:#78350f; color:#fef3c7; }
    .risk-pill.low    { background:#14532d; color:#dcfce7; }

    /* Info / explanation cards */
    .info-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .info-card h4 { margin-top: 0; color: #6366f1 !important; }

    /* Footer */
    .app-footer {
        margin-top: 40px;
        padding: 18px;
        text-align: center;
        color: #94a3b8;
        border-top: 1px solid #334155;
        font-size: 0.9rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_scoring_df():
    if not SCORING_CSV.exists():
        return None
    df = pd.read_csv(SCORING_CSV)
    return df


@st.cache_data(show_spinner=False)
def load_recommendations_df():
    if not RECOMMENDATIONS_CSV.exists():
        return None
    return pd.read_csv(RECOMMENDATIONS_CSV)


@st.cache_data(show_spinner=False)
def load_raw_data():
    if not RAW_DATA_CSV.exists():
        return None
    return pd.read_csv(RAW_DATA_CSV)


@st.cache_resource(show_spinner=False)
def load_models():
    """Load all three trained models (and the best). Missing ones are skipped."""
    loaded = {}
    for name, path in MODEL_PATHS.items():
        if path.exists():
            try:
                loaded[name] = joblib.load(path)
            except Exception:
                pass
    best = None
    if BEST_MODEL_PATH.exists():
        try:
            best = joblib.load(BEST_MODEL_PATH)
        except Exception:
            best = None
    return loaded, best


@st.cache_data(show_spinner=False)
def compute_model_metrics():
    """
    Best-effort metrics table for Page 2.

    model_trainer.py saves the models but not the results dict, so we
    rebuild the test split here (mirroring preprocessor + feature_engineer
    behavior) and predict with each saved model. If anything fails, we
    return None and Page 2 falls back to showing the saved comparison PNG.
    """
    try:
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
        )
        from src.feature_engineer import engineer_features
        from src.preprocessor import preprocess_data

        raw_df = load_raw_data()
        if raw_df is None:
            return None

        df_feat = engineer_features(raw_df)
        X_train, X_test, y_train, y_test, _scaler, _features, _processed = preprocess_data(df_feat)

        # Same NaN scrub as model_trainer.
        X_test = pd.DataFrame(np.nan_to_num(X_test, nan=0.0),
                              columns=X_test.columns if hasattr(X_test, "columns") else None)

        models, _ = load_models()
        if not models:
            return None

        rows = {}
        confusion_data = {}
        for name, model in models.items():
            try:
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]
                rows[name] = {
                    "AUC-ROC":   roc_auc_score(y_test, y_proba),
                    "Precision": precision_score(y_test, y_pred),
                    "Recall":    recall_score(y_test, y_pred),
                    "F1":        f1_score(y_test, y_pred),
                    "Accuracy":  accuracy_score(y_test, y_pred),
                }
                confusion_data[name] = {"y_test": y_test, "y_pred": y_pred}
            except Exception:
                continue

        if not rows:
            return None

        return {
            "table": pd.DataFrame(rows).T.round(4),
            "confusion": confusion_data,
        }
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_metrics_table():
    """
    Convenience alias around compute_model_metrics() that returns just the
    metrics DataFrame (or None). Kept so other parts of the app can ask
    for "the table" without knowing about the larger metrics pack.
    """
    pack = compute_model_metrics()
    if pack is None:
        return None
    return pack.get("table")


@st.cache_data(show_spinner=False)
def compute_shap_top_features(top_n: int = 15):
    """
    Compute mean absolute SHAP values for the saved best model and return
    the top N features as a DataFrame. Cached so the SHAP page is instant
    after the first load.

    Returns
    -------
    pandas.DataFrame | None
        DataFrame with columns ["feature", "importance"], sorted descending,
        or None if any step (loading models, computing SHAP) fails.
    """
    try:
        from src.feature_engineer import engineer_features
        from src.preprocessor import preprocess_data
        from src.explainer import generate_shap_values

        raw_df = load_raw_data()
        if raw_df is None:
            return None

        df_feat = engineer_features(raw_df)
        X_train, X_test, _y_train, _y_test, _scaler, feature_names, _processed = preprocess_data(df_feat)

        # Same NaN scrub as model_trainer.
        X_train = pd.DataFrame(np.nan_to_num(X_train, nan=0.0),
                               columns=X_train.columns if hasattr(X_train, "columns") else feature_names)
        X_test = pd.DataFrame(np.nan_to_num(X_test, nan=0.0),
                              columns=X_test.columns if hasattr(X_test, "columns") else feature_names)

        _, best_model = load_models()
        if best_model is None:
            return None

        # Pick the matching model_type tag for the explainer.
        cls_name = type(best_model).__name__.lower()
        if "xgb" in cls_name:
            model_type = "xgboost"
        elif "lgbm" in cls_name or "lightgbm" in cls_name:
            model_type = "lightgbm"
        else:
            model_type = "logistic"

        shap_values, _ = generate_shap_values(
            best_model, X_train, X_test, feature_names, model_type
        )
        mean_abs = np.abs(np.asarray(shap_values)).mean(axis=0)

        return (
            pd.DataFrame({"feature": feature_names, "importance": mean_abs})
            .sort_values("importance", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )
    except Exception:
        return None


def plotly_shap_summary(top_features_df):
    """
    Compact, interactive horizontal bar chart of top SHAP features.
    Capped at 400px tall with smaller axis fonts so the chart fits
    cleanly inside the dashboard layout.
    """
    fig = px.bar(
        top_features_df.iloc[::-1],  # reverse for top-down ranking
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale="Viridis",
        title="Top 15 Churn Drivers (SHAP Feature Importance)",
    )
    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>"
    )
    fig.update_layout(
        height=400,
        font=dict(size=11),
        xaxis_title="Mean |SHAP value|",
        yaxis_title=None,
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig.update_xaxes(tickfont=dict(size=11))
    fig.update_yaxes(tickfont=dict(size=11))
    return _style_plotly(fig, height=400)


# ----------------------------------------------------------------------
# Small UI helpers
# ----------------------------------------------------------------------
def show_image_or_warn(path: Path, caption: str = ""):
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"Missing artifact: `{path.name}`. Run the pipeline to generate it.")


def risk_pill_html(level: str) -> str:
    cls = level.lower() if level in ("HIGH", "MEDIUM", "LOW") else "low"
    return f'<span class="risk-pill {cls}">{level}</span>'


# ----------------------------------------------------------------------
# Plotly chart builders
# ----------------------------------------------------------------------
# We use a shared dark template + transparent paper background so the
# Plotly charts blend with the app's own dark theme. Centralizing the
# styling here keeps the charts visually consistent across pages.
PLOTLY_TEMPLATE = "plotly_dark"
RISK_COLORS = {"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#22c55e"}
CHURN_COLORS = {"Yes": "#ef4444", "No": "#22c55e"}


def _style_plotly(fig, height=420):
    """Apply common look-and-feel to every Plotly figure in this app."""
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        margin=dict(l=10, r=10, t=60, b=10),
        height=height,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        # Disable click-and-drag zoom so the cursor doesn't turn into
        # the plotly "+" reticle, and tame hover behavior for clean
        # tooltips that follow the data, not the pointer.
        dragmode=False,
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#1e293b",
            bordercolor="#6366f1",
            font=dict(color="#f8fafc"),
        ),
    )
    # Belt-and-braces: clamp x/y axes against accidental drag-zoom.
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


# Plotly chart config shared by every st.plotly_chart() call.
# Hides the modebar (zoom/pan/save buttons) and disables image download
# so the charts feel like dashboard widgets instead of analysis tools.
PLOTLY_CONFIG = {
    "displayModeBar": False,
    "displaylogo": False,
    "staticPlot": False,  # keep hover tooltips alive
    "scrollZoom": False,
}


def plotly_churn_distribution(df):
    """Interactive bar chart of Churn Yes / No counts with % labels."""
    counts = (
        df["Churn"]
        .value_counts()
        .rename_axis("Churn")
        .reset_index(name="count")
    )
    total = counts["count"].sum()
    counts["percent"] = (counts["count"] / total * 100).round(1)
    counts["label"] = counts.apply(
        lambda r: f"{int(r['count'])} ({r['percent']}%)", axis=1
    )

    fig = px.bar(
        counts,
        x="Churn",
        y="count",
        color="Churn",
        text="label",
        color_discrete_map=CHURN_COLORS,
        title="Customer Churn Distribution",
        hover_data={"Churn": True, "count": True, "percent": ":.1f", "label": False},
    )
    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>Churn: %{x}</b><br>Customers: %{y}<extra></extra>",
    )
    fig.update_layout(showlegend=False, yaxis_title="Number of customers")
    return _style_plotly(fig)


def plotly_risk_segmentation(scoring_df):
    """Interactive pie chart of HIGH / MEDIUM / LOW risk customers."""
    order = ["HIGH", "MEDIUM", "LOW"]
    counts = (
        scoring_df["risk_level"]
        .value_counts()
        .reindex(order)
        .fillna(0)
        .astype(int)
        .rename_axis("risk_level")
        .reset_index(name="count")
    )
    counts = counts[counts["count"] > 0]

    fig = px.pie(
        counts,
        names="risk_level",
        values="count",
        color="risk_level",
        color_discrete_map=RISK_COLORS,
        title="Customer Risk Segmentation",
        hole=0.4,
    )
    fig.update_traces(
        textinfo="label+percent",
        textfont=dict(size=14, color="#f8fafc"),
        marker=dict(line=dict(color="#0f172a", width=2)),
        hovertemplate="<b>%{label} risk</b><br>Customers: %{value}<br>"
                      "Share: %{percent}<extra></extra>",
    )
    return _style_plotly(fig)


def plotly_churn_probability_distribution(scoring_df):
    """Interactive histogram with vertical lines at the 0.40 / 0.70 thresholds."""
    fig = px.histogram(
        scoring_df,
        x="churn_probability",
        nbins=30,
        color_discrete_sequence=["#6366f1"],
        title="Churn Probability Distribution",
    )
    fig.update_traces(
        marker_line_color="#0f172a",
        marker_line_width=1,
        hovertemplate="<b>Probability bin:</b> %{x}<br>"
                      "Customers: %{y}<extra></extra>",
    )
    fig.add_vline(
        x=0.40, line_dash="dash", line_color="#f59e0b", line_width=2,
        annotation_text="MEDIUM (0.40)", annotation_position="top",
        annotation_font_color="#f59e0b",
    )
    fig.add_vline(
        x=0.70, line_dash="dash", line_color="#ef4444", line_width=2,
        annotation_text="HIGH (0.70)", annotation_position="top",
        annotation_font_color="#ef4444",
    )
    fig.update_layout(xaxis_title="Churn probability", yaxis_title="Number of customers",
                      bargap=0.05)
    return _style_plotly(fig)


def plotly_model_comparison(metrics_table):
    """Grouped bar chart comparing AUC-ROC / F1 / Precision / Recall per model."""
    metric_cols = [c for c in ["AUC-ROC", "F1", "Precision", "Recall"]
                   if c in metrics_table.columns]
    long_df = (
        metrics_table[metric_cols]
        .reset_index()
        .rename(columns={"index": "Model"})
        .melt(id_vars="Model", var_name="Metric", value_name="Score")
    )

    fig = px.bar(
        long_df,
        x="Model", y="Score", color="Metric", barmode="group",
        text=long_df["Score"].round(3),
        color_discrete_sequence=["#3498db", "#9b59b6", "#f39c12", "#1abc9c"],
        title="Model Performance Comparison",
        hover_data={"Score": ":.4f"},
    )
    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{y:.4f}<extra></extra>",
    )
    fig.update_layout(yaxis=dict(range=[0, 1.05]), legend_title_text="Metric")
    return _style_plotly(fig, height=480)


# ======================================================================
# Top horizontal navigation bar
# ======================================================================
# We render the nav as a row of equal-width buttons inside a styled
# container. The active page is tracked in session state so reruns
# (button clicks, filter changes, etc.) stay on the same page. The
# active button is given Streamlit's "primary" type so the navbar CSS
# can highlight it with the indigo accent color.
#
# PAGES is the canonical list of page IDs (used by the router).
# NAV_LABELS maps each page ID to a shorter label shown on the button,
# so we can keep the routing keys stable even after a rename.
PAGES = [
    "Overview",
    "Model Performance",
    "Risk Segmentation",
    "Feature Importance & SHAP",
    "Retention Recommendations",
    "Customer Drill-Down",
]
NAV_LABELS = {
    "Overview":                  "Overview",
    "Model Performance":         "Model Performance",
    "Risk Segmentation":         "Risk Segmentation",
    "Feature Importance & SHAP": "SHAP",
    "Retention Recommendations": "Recommendations",
    "Customer Drill-Down":       "Drill-Down",
}

if "active_page" not in st.session_state:
    st.session_state.active_page = PAGES[0]


def _set_page(page_name: str):
    """Callback fired when a navbar button is clicked."""
    st.session_state.active_page = page_name


# Wrap the navbar in a div so our scoped CSS only styles these buttons,
# not every button on the page.
nav_cols = st.columns(len(PAGES), gap="small")
for col, page_name in zip(nav_cols, PAGES):
    with col:
        is_active = (st.session_state.active_page == page_name)
        st.button(
            NAV_LABELS[page_name],
            key=f"nav_{page_name}",
            type="primary" if is_active else "secondary",
            on_click=_set_page,
            args=(page_name,),
            use_container_width=True,
        )

page = st.session_state.active_page


# ======================================================================
# PAGE 1 - Overview
# ======================================================================
def page_overview():
    st.title("Customer Churn Analytics Dashboard")
    st.caption("A single-pane view of customer churn risk and retention priorities.")

    try:
        scoring_df = load_scoring_df()
        if scoring_df is None or len(scoring_df) == 0:
            st.error("`outputs/weekly_scoring_week_1.csv` not found. Run `python main.py` first.")
            return

        total_customers = len(scoring_df)
        high_count = int((scoring_df["risk_level"] == "HIGH").sum())
        churn_rate = (scoring_df["churn_probability"] >= 0.5).mean() * 100
        avg_prob = scoring_df["churn_probability"].mean()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Customers Scored", f"{total_customers:,}")
        with c2:
            st.metric("HIGH Risk Customers", f"{high_count:,}",
                      delta=f"{(high_count/total_customers)*100:.1f}% of base",
                      delta_color="inverse")
        with c3:
            st.metric("Predicted Churn Rate", f"{churn_rate:.2f}%")
        with c4:
            st.metric("Avg Churn Probability", f"{avg_prob:.4f}")

        st.markdown("")
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Churn Distribution")
            raw_df = load_raw_data()
            if raw_df is not None and "Churn" in raw_df.columns:
                st.plotly_chart(
                    plotly_churn_distribution(raw_df),
                    use_container_width=True,
                    config=PLOTLY_CONFIG,
                )
            else:
                # Fallback to the saved PNG if the raw CSV is missing.
                show_image_or_warn(PLOT_CHURN_DIST)
        with col_right:
            st.subheader("Risk Segmentation")
            st.plotly_chart(
                plotly_risk_segmentation(scoring_df),
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )

        st.markdown(
            """
            <div class="info-card">
              <h4>What is churn, in business terms?</h4>
              <p>Churn happens when a customer stops doing business with the company.
              For a subscription business it is the single most expensive event you
              can have: acquiring a new customer typically costs 5–7x more than
              keeping an existing one. This dashboard ranks customers by how likely
              they are to leave, so the retention team can act on the riskiest
              accounts first instead of treating every customer the same.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    except Exception as e:
        st.error(f"Could not render the Overview page: {e}")


# ======================================================================
# PAGE 2 - Model Performance
# ======================================================================
def page_model_performance():
    st.title("Model Performance")
    st.caption("How well each trained model separates churners from loyal customers.")

    try:
        metrics_pack = compute_model_metrics()

        if metrics_pack is not None:
            st.subheader("Metric Comparison Table")
            st.dataframe(metrics_pack["table"], use_container_width=True)

            st.subheader("Model Performance Comparison")
            st.plotly_chart(
                plotly_model_comparison(metrics_pack["table"]),
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )
        else:
            st.warning(
                "Could not compute the live metric table (saved models or raw "
                "data unavailable). Showing the saved comparison chart instead."
            )
            st.subheader("Model Performance Comparison")
            show_image_or_warn(PLOT_MODEL_COMPARISON)

        # Confusion matrix for the best model.
        st.subheader("Confusion Matrix — Best Model")
        _, best_model = load_models()
        best_name = None
        if metrics_pack is not None and not metrics_pack["table"].empty:
            best_name = metrics_pack["table"]["AUC-ROC"].idxmax()

        if best_name is None:
            # Fallback: try XGBoost first since it's usually the best on this data.
            for candidate in ("XGBoost", "LightGBM", "Logistic Regression"):
                slug = candidate.lower().replace(" ", "_")
                fp = OUTPUTS_DIR / f"confusion_matrix_{slug}.png"
                if fp.exists():
                    best_name = candidate
                    show_image_or_warn(fp, caption=f"Confusion matrix — {candidate}")
                    break
            if best_name is None:
                st.warning("No confusion matrix images found in outputs/.")
        else:
            slug = best_name.lower().replace(" ", "_")
            fp = OUTPUTS_DIR / f"confusion_matrix_{slug}.png"
            show_image_or_warn(fp, caption=f"Confusion matrix — {best_name}")
            st.success(f"Best model selected by AUC-ROC: **{best_name}**")

        # Plain-language metric explanations.
        st.markdown("### What do these metrics mean?")
        st.markdown(
            """
            <div class="info-card">
              <h4>AUC-ROC</h4>
              <p>How well the model separates churners from non-churners across
              all possible thresholds. Closer to 1.0 is better. This is the
              most important metric for ranking customers by risk.</p>
            </div>
            <div class="info-card">
              <h4>Recall</h4>
              <p>Out of all customers who actually churned, how many did we catch?
              For churn this is the metric that maps to lost revenue — missed
              churners walk out the door.</p>
            </div>
            <div class="info-card">
              <h4>Precision</h4>
              <p>Out of customers we predicted would churn, how many actually did?
              High precision means fewer wasted retention offers on customers who
              would have stayed anyway.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    except Exception as e:
        st.error(f"Could not render the Model Performance page: {e}")


# ======================================================================
# PAGE 3 - Risk Segmentation
# ======================================================================
def page_risk_segmentation():
    st.title("Risk Segmentation")
    st.caption("Filter and prioritize customers by predicted risk tier.")

    try:
        scoring_df = load_scoring_df()
        if scoring_df is None or len(scoring_df) == 0:
            st.error("`outputs/weekly_scoring_week_1.csv` not found.")
            return

        # Risk tier counts.
        counts = scoring_df["risk_level"].value_counts().reindex(["HIGH", "MEDIUM", "LOW"]).fillna(0).astype(int)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("HIGH risk",   f"{int(counts.get('HIGH', 0)):,}")
        with c2:
            st.metric("MEDIUM risk", f"{int(counts.get('MEDIUM', 0)):,}")
        with c3:
            st.metric("LOW risk",    f"{int(counts.get('LOW', 0)):,}")

        # Filters.
        st.markdown("### Filter Customers")
        f1, f2 = st.columns([2, 1])
        with f1:
            risk_filter = st.multiselect(
                "Filter by risk level",
                options=["HIGH", "MEDIUM", "LOW"],
                default=["HIGH", "MEDIUM", "LOW"],
            )
        with f2:
            sort_desc = st.checkbox("Sort by churn probability (desc)", value=True)

        view = scoring_df[scoring_df["risk_level"].isin(risk_filter)].copy()
        if sort_desc:
            view = view.sort_values("churn_probability", ascending=False)

        # Default Streamlit table styling (no custom row colors).
        st.dataframe(
            view,
            use_container_width=True,
            height=420,
            column_config={
                "churn_probability": st.column_config.NumberColumn(format="%.4f"),
            },
        )

        st.subheader("Churn Probability Distribution")
        st.plotly_chart(
            plotly_churn_probability_distribution(scoring_df),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

    except Exception as e:
        st.error(f"Could not render the Risk Segmentation page: {e}")


# ======================================================================
# PAGE 4 - Feature Importance & SHAP
# ======================================================================
def page_feature_importance():
    st.title("Feature Importance & SHAP")
    st.caption("Which features drive the model's churn predictions, and why.")

    try:
        st.subheader("Top Churn Drivers (SHAP)")
        shap_top = compute_shap_top_features(top_n=15)
        if shap_top is not None and len(shap_top) > 0:
            st.plotly_chart(
                plotly_shap_summary(shap_top),
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )
        else:
            # Fall back to the saved PNG, sized down to match the new
            # compact look so it doesn't dominate the page.
            if PLOT_SHAP_SUMMARY.exists():
                st.image(str(PLOT_SHAP_SUMMARY), width=700)
            else:
                st.warning(
                    "Missing artifact: `shap_summary.png`. Run the pipeline to generate it."
                )

        st.subheader("Feature Correlation Heatmap")
        if PLOT_CORRELATION_HEATMAP.exists():
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(str(PLOT_CORRELATION_HEATMAP), use_column_width=True)
        else:
            st.warning(
                "Missing artifact: `correlation_heatmap.png`. Run the pipeline to generate it."
            )

        st.markdown("### Top 5 Churn Drivers, in Plain English")
        drivers = [
            ("Contract type",
             "Month-to-month contracts are the single strongest churn signal. "
             "Customers with no commitment can leave on the next billing cycle "
             "with no penalty, so they do — at much higher rates than annual "
             "or two-year contract holders."),
            ("Tenure",
             "Newer customers churn far more than long-tenured ones. The first "
             "12 months are the danger zone: customers haven't built loyalty "
             "yet and a single bad experience can push them out."),
            ("Monthly charges",
             "Customers paying high monthly bills are more sensitive to value. "
             "If they don't feel they're getting their money's worth, they "
             "shop around. High monthly charges combined with short tenure is "
             "an especially risky pattern."),
            ("Tech support / Online security",
             "Customers without these add-on services feel unsupported when "
             "something goes wrong. A single bad incident with no support "
             "channel often becomes the trigger event for churn."),
            ("Internet service (Fiber optic)",
             "Fiber optic customers churn more in this dataset, which usually "
             "reflects either pricing pressure from competitors or unresolved "
             "service-quality complaints. Worth investigating field reports."),
        ]
        for title, body in drivers:
            st.markdown(
                f'<div class="info-card"><h4>{title}</h4><p>{body}</p></div>',
                unsafe_allow_html=True,
            )

    except Exception as e:
        st.error(f"Could not render the Feature Importance page: {e}")


# ======================================================================
# PAGE 5 - Retention Recommendations
# ======================================================================
def page_recommendations():
    st.title("Retention Recommendations")
    st.caption("Concrete actions for the retention team, grouped by risk reason.")

    try:
        rec_df = load_recommendations_df()
        if rec_df is None or len(rec_df) == 0:
            st.warning("`outputs/retention_recommendations.csv` not found or empty.")
            return

        total_high = len(rec_df)

        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Customers needing immediate action", f"{total_high:,}")
        with c2:
            st.markdown(
                f"""
                <div class="info-card">
                  <h4>Business summary</h4>
                  <p><b>{total_high} customers need immediate retention action.</b>
                  Each row below pairs a customer with the top reason they are at
                  risk and the recommended next step. Filter by risk level or
                  recommendation type to plan the week's outreach.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Filters.
        f1, f2 = st.columns(2)
        with f1:
            level_options = sorted(rec_df["risk_level"].unique().tolist())
            level_filter = st.multiselect(
                "Filter by risk level", options=level_options, default=level_options
            )
        with f2:
            action_options = sorted(rec_df["recommended_action"].unique().tolist())
            action_filter = st.multiselect(
                "Filter by recommended action",
                options=action_options,
                default=action_options,
            )

        view = rec_df[
            rec_df["risk_level"].isin(level_filter)
            & rec_df["recommended_action"].isin(action_filter)
        ].copy()

        # Default Streamlit table styling (no custom row colors).
        st.dataframe(
            view,
            use_container_width=True,
            height=420,
            column_config={
                "churn_probability": st.column_config.NumberColumn(format="%.4f"),
            },
        )

        # Recommendation type breakdown.
        st.subheader("Recommendations by Type")
        breakdown = (
            rec_df["recommended_action"]
            .value_counts()
            .rename_axis("Recommended action")
            .reset_index(name="Customer count")
        )
        st.dataframe(breakdown, use_container_width=True)

    except Exception as e:
        st.error(f"Could not render the Recommendations page: {e}")


# ======================================================================
# PAGE 6 - Customer Drill-Down
# ======================================================================
def page_drilldown():
    st.title("Customer Drill-Down")
    st.caption("Inspect a single customer's features, risk, reasons, and recommended action.")

    try:
        scoring_df = load_scoring_df()
        rec_df = load_recommendations_df()
        raw_df = load_raw_data()

        if scoring_df is None or len(scoring_df) == 0:
            st.error("`outputs/weekly_scoring_week_1.csv` not found.")
            return

        max_idx = int(scoring_df["customer_index"].max())
        idx = st.number_input(
            f"Enter customer index (0 to {max_idx})",
            min_value=0, max_value=max_idx, value=0, step=1,
        )

        # Pull this customer's scoring row.
        row = scoring_df[scoring_df["customer_index"] == idx]
        if row.empty:
            st.warning(f"No scoring record found for customer_index={idx}.")
            return
        row = row.iloc[0]

        risk_level = row["risk_level"]
        churn_prob = float(row["churn_probability"])

        # Top metric strip.
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Customer index", int(idx))
        with c2:
            st.metric("Risk level", risk_level)
        with c3:
            st.metric("Churn probability", f"{churn_prob:.4f}")

        st.markdown(
            f'<p>Risk tier: {risk_pill_html(risk_level)}</p>',
            unsafe_allow_html=True,
        )

        # Pull the top reasons + action from recommendations (HIGH only).
        rec_row = None
        if rec_df is not None:
            rec_match = rec_df[rec_df["customer_index"] == idx]
            if not rec_match.empty:
                rec_row = rec_match.iloc[0]

        st.subheader("Top 3 Churn Reasons")
        if rec_row is not None:
            reasons_html = "<ul>"
            for col in ["top_reason_1", "top_reason_2", "top_reason_3"]:
                val = rec_row.get(col, "")
                if isinstance(val, str) and val:
                    reasons_html += f"<li><b>{val}</b></li>"
            reasons_html += "</ul>"
            st.markdown(
                f'<div class="info-card">{reasons_html}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info(
                "No SHAP-based reasons stored for this customer. "
                "Per-customer reasons are generated only for HIGH-risk customers "
                "in `retention_recommendations.csv`."
            )

        st.subheader("Recommended Action")
        if rec_row is not None and "recommended_action" in rec_row:
            st.markdown(
                f'<div class="info-card"><h4>{rec_row["recommended_action"]}</h4></div>',
                unsafe_allow_html=True,
            )
        else:
            if risk_level == "HIGH":
                st.markdown(
                    '<div class="info-card"><h4>Schedule retention call within 48 hours</h4></div>',
                    unsafe_allow_html=True,
                )
            elif risk_level == "MEDIUM":
                st.markdown(
                    '<div class="info-card"><h4>Add to nurture campaign and monitor next week</h4></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="info-card"><h4>No action required. Continue routine engagement.</h4></div>',
                    unsafe_allow_html=True,
                )

        # All raw features for this customer.
        st.subheader("All Customer Features")
        if raw_df is not None and 0 <= idx < len(raw_df):
            cust_row = raw_df.iloc[int(idx)]
            features_df = (
                pd.DataFrame({"Feature": cust_row.index, "Value": cust_row.values})
                .reset_index(drop=True)
            )
            st.dataframe(features_df, use_container_width=True, height=520)
        else:
            st.warning(
                "Raw dataset not available, so the full feature view cannot be shown. "
                "Place the CSV at `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`."
            )

    except Exception as e:
        st.error(f"Could not render the Drill-Down page: {e}")


# ======================================================================
# Router
# ======================================================================
PAGE_FUNCS = {
    "Overview":                    page_overview,
    "Model Performance":           page_model_performance,
    "Risk Segmentation":           page_risk_segmentation,
    "Feature Importance & SHAP":   page_feature_importance,
    "Retention Recommendations":   page_recommendations,
    "Customer Drill-Down":         page_drilldown,
}

PAGE_FUNCS[page]()


# ======================================================================
# Footer
# ======================================================================
st.markdown(
    '<div class="app-footer">Customer Churn Prediction System | '
    'Teyzix Core Internship ML-3 | Built with XGBoost + SHAP</div>',
    unsafe_allow_html=True,
)
