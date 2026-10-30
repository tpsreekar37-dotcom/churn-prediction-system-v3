"""
Business Analytics System — AI-Driven Customer Churn Prediction
Alabaster Sovereign White & Gold Edition (Specification: white_gold.md)
Run: streamlit run dashboard/app.py
"""
import json
import os
import sys
import time
import datetime
from html import escape

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.predict import load_model, predict_batch, predict_single
from src.generic_pipeline import predict_generic_batch, run_generic_pipeline
from src.sql_integration import store_predictions_to_sqlite, run_analytics, get_latest_run_id
from src.utils import (
    DATA_PROCESSED, MODELS_DIR, DECISION_THRESHOLD,
    classify_risk, classify_risk_emoji, is_telco_schema,
    find_col as _find_col,
    safe_float as _safe_float,
    safe_int as _safe_int,
    get_revenue_metrics as _get_revenue_metrics,
    find_revenue_column,
)
from src.model_registry import (
    available_models as _available_models,
    load_model_by_name,
    select_best_model,
    run_predictions,
    evaluate_predictions,
    validate_dataset,
    load_saved_model_results,
    canonical_metric_rows,
    metric_rows,
    MODEL_DESCRIPTIONS,
)

# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Business Analytics System — AI Customer Churn Prediction",
    page_icon="⚜",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# FILE READER — strips Excel phantom rows/columns immediately on load
# ─────────────────────────────────────────────────────────────────────────────

def _read_uploaded_file(uploaded_file) -> pd.DataFrame:
    """
    Read a CSV or Excel upload and immediately drop entirely-empty rows/columns
    that Excel adds as ghost formatting artifacts. Also resets the index.
    """
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    # Drop columns that are 100% NaN (Excel ghost columns)
    df = df.dropna(axis=1, how="all")
    # Drop rows that are 100% NaN (Excel ghost rows at the bottom)
    df = df.dropna(axis=0, how="all")
    df = df.reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# DESIGN TOKENS — Alabaster Sovereign
# ═══════════════════════════════════════════════════════════════════════════════
T = {
    "bg":           "#FAFAF7",
    "card":         "#FFFFFF",
    "card_hover":   "#FDFDFB",
    "surface_sub":  "#F4F4EE",
    "sidebar":      "#F7F7F2",
    "gold":         "#C59B27",
    "gold_deep":    "#8F6700",
    "gold_radiant": "#D4AF37",
    "gold_light":   "rgba(197, 155, 39, 0.12)",
    "gold_glow":    "rgba(197, 155, 39, 0.3)",
    "border":       "#DDC8A4",
    "border_sub":   "#EADDC7",
    "border_hover": "#C59B27",
    "text":         "#000000",
    "text_sec":     "#000000",
    "text_muted":   "#111111",
    "text_faint":   "#222222",
    "green":        "#047857",
    "green_bg":     "#ECFDF5",
    "green_border": "rgba(4, 120, 87, 0.5)",
    "amber":        "#B45309",
    "amber_bg":     "#FFFBEB",
    "amber_border": "rgba(180, 83, 9, 0.5)",
    "red":          "#B91C1C",
    "red_bg":       "#FEF2F2",
    "red_border":   "rgba(185, 28, 28, 0.5)",
}

# ═══════════════════════════════════════════════════════════════════════════════
# CSS — load style.css + inject core overrides (unchanged from original)
# ═══════════════════════════════════════════════════════════════════════════════
css_file_path = os.path.join(os.path.dirname(__file__), "style.css")
custom_css = ""
if os.path.exists(css_file_path):
    with open(css_file_path, "r", encoding="utf-8") as f:
        custom_css = f.read()

st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@600;700;800&family=Noto+Serif:ital,wght@0,600;0,700;0,800;1,600&display=swap');
{custom_css}
.stApp {{background-color:{T['bg']} !important;color:#000000 !important;font-family:'Manrope',-apple-system,BlinkMacSystemFont,sans-serif !important;-webkit-font-smoothing:subpixel-antialiased;}}
.stApp,.stApp p,.stApp span,.stApp label,.stApp div,.stApp li,.stApp b,.stApp strong,.stApp a,[data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] span,[data-testid="stMarkdownContainer"] li,[data-testid="stMarkdownContainer"] strong,[data-testid="stWidgetLabel"] p,[data-testid="stWidgetLabel"] label,[data-testid="stWidgetLabel"] span {{color:#000000 !important;font-weight:700 !important;opacity:1 !important;}}
h1,h2,h3,h4,h5,h6,.page-title,.brand-title,.section-title {{font-family:'Noto Serif',Georgia,serif !important;color:#000000 !important;font-weight:800 !important;opacity:1 !important;}}
.tabular-nums,.kpi-value,[data-testid="stMetricValue"] {{font-variant-numeric:tabular-nums !important;font-weight:800 !important;}}
.stButton>button {{background:linear-gradient(180deg,{T['gold_radiant']} 0%,{T['gold']} 100%) !important;color:#FFFFFF !important;border:1.5px solid #805B00 !important;border-radius:4px !important;font-family:'Manrope',sans-serif !important;font-weight:800 !important;font-size:0.92rem !important;padding:0.55rem 1.4rem !important;letter-spacing:0.01em !important;text-shadow:0 1px 2px rgba(0,0,0,0.35) !important;box-shadow:0 2px 6px rgba(0,0,0,0.12),inset 0 1px 0 rgba(255,255,255,0.35) !important;transition:all 0.2s cubic-bezier(0.16,1,0.3,1) !important;}}
.stButton>button * {{color:#FFFFFF !important;font-weight:800 !important;}}
.stButton>button:hover {{background:linear-gradient(180deg,#E6C759 0%,{T['gold']} 100%) !important;border-color:#6B4900 !important;box-shadow:0 4px 16px 0 {T['gold_glow']} !important;transform:translateY(-1px);}}
.stButton>button:active {{background:#805B00 !important;transform:translateY(0);}}
.stDownloadButton>button {{background:#FFFFFF !important;color:#000000 !important;border:1.5px solid {T['border']} !important;border-radius:4px !important;font-family:'Manrope',sans-serif !important;font-weight:800 !important;font-size:0.92rem !important;padding:0.55rem 1.4rem !important;transition:all 0.2s ease !important;}}
.stDownloadButton>button * {{color:#000000 !important;font-weight:800 !important;}}
.stDownloadButton>button:hover {{border-color:{T['gold']} !important;color:#6B4900 !important;background:{T['bg']} !important;box-shadow:0 2px 8px rgba(197,155,39,0.25) !important;}}
[data-testid="stSidebar"] {{background:{T['sidebar']} !important;border-right:1.5px solid {T['border']} !important;}}
[data-testid="stSidebar"] * {{font-family:'Manrope',sans-serif !important;color:#000000 !important;font-weight:700 !important;}}
[data-testid="stDataFrame"] {{border:1px solid {T['border']} !important;border-radius:8px !important;background:#FFFFFF !important;box-shadow:0 1px 3px 0 rgba(30,41,59,0.03) !important;}}
div[data-baseweb="select"]>div,input[type="text"],input[type="number"] {{background:#FFFFFF !important;border:1px solid {T['border']} !important;border-radius:4px !important;color:{T['text']} !important;}}
div[data-baseweb="select"]>div:focus-within,input[type="text"]:focus,input[type="number"]:focus {{border-color:{T['gold']} !important;box-shadow:0 0 0 3px rgba(197,155,39,0.15) !important;}}
</style>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# REUSABLE UI COMPONENTS (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

def page_header(badge: str, title: str, desc: str):
    st.markdown(f"""
    <div style="margin-bottom:1.4rem;">
        <div class="section-badge">{escape(badge)}</div>
        <div class="page-title">{escape(title)}</div>
        <div class="page-desc">{escape(desc)}</div>
    </div>
    """, unsafe_allow_html=True)


def kpi_card(col, icon: str, label: str, value: str, desc: str = "", color: str = ""):
    accent = color or T["text"]
    with col:
        st.markdown(f"""<div class="kpi-card">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:0.25rem;">
                <div class="kpi-label">{escape(label)}</div>
                <div class="kpi-icon">{icon}</div>
            </div>
            <div class="kpi-value" style="color:{accent} !important;">{escape(str(value))}</div>
            <div class="kpi-desc">{escape(desc)}</div>
        </div>""", unsafe_allow_html=True)


def info_box(text: str):
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)


def section_heading(title: str):
    st.markdown(f"""
    <h3 style="font-family:'Noto Serif',Georgia,serif;font-size:1.35rem;font-weight:800;color:#000000 !important;margin:1.6rem 0 0.8rem;letter-spacing:-0.01em;">
        {escape(title)}
    </h3>
    """, unsafe_allow_html=True)


def chart_style(fig, h=340):
    fig.update_layout(
        height=h, paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        margin=dict(l=16, r=16, t=44, b=16),
        font=dict(color=T["text"], size=12, family="Manrope, sans-serif"),
        legend=dict(orientation="h", y=1.06,
                    font=dict(color=T["text"], size=11, family="Manrope, sans-serif"),
                    bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=T["border_hover"],
                        font_size=12, font_color=T["text"], font_family="Manrope, sans-serif"),
    )
    fig.update_xaxes(gridcolor=T["border_sub"],
                     tickfont=dict(color=T["text"], size=11, family="Manrope, sans-serif"),
                     title_font=dict(color=T["text"], size=12, family="Manrope, sans-serif"),
                     linecolor=T["border"], zerolinecolor=T["border"])
    fig.update_yaxes(gridcolor=T["border_sub"],
                     tickfont=dict(color=T["text"], size=11, family="Manrope, sans-serif"),
                     title_font=dict(color=T["text"], size=12, family="Manrope, sans-serif"),
                     linecolor=T["border"], zerolinecolor=T["border"])
    return fig


def find_col(df, candidates, default=None, exact=False):
    return _find_col(df, candidates, default=default, exact=exact)


def safe_float(v, default=0.0):
    return _safe_float(v, default)


def safe_int(v, default=0):
    return _safe_int(v, default)


def get_revenue_metrics(predicted, sql_results=None):
    return _get_revenue_metrics(predicted, sql_results)


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED BUSINESS LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def risk_segment_summaries(predicted, target_column=None):
    excluded = {"Churn Probability", "Churn Prediction", "Risk Level"}
    segments = []
    for column in predicted.columns:
        if column in excluded or column == target_column or "id" in column.lower() \
                or predicted[column].nunique(dropna=False) < 2 or predicted[column].nunique(dropna=False) > 12:
            continue
        grouped = predicted.groupby(column, dropna=False)["Churn Probability"].agg(["mean", "count"]) \
            .sort_values("mean", ascending=False).head(1)
        if not grouped.empty:
            value = grouped.index[0]
            segments.append(
                f"{column} = '{value}' ({grouped.iloc[0]['mean']:.1f}% avg risk, "
                f"{int(grouped.iloc[0]['count'])} customers)"
            )
    return segments


# ═══════════════════════════════════════════════════════════════════════════════
# CENTRAL PREDICTION RUNNER  (single function used by ALL pages)
# ═══════════════════════════════════════════════════════════════════════════════

def _run_and_store_predictions(df: pd.DataFrame, model_name: str | None = None) -> pd.DataFrame | None:
    """
    Score ``df`` with the given model (or auto-selected best), store to SQLite,
    and return the predictions DataFrame. Returns None on failure.
    Sets st.session_state keys:
        uploaded_predictions, upload_error, latest_run_id, active_model_name,
        active_model_selection_reason, active_model_metric_value
    """
    if df is None or df.empty:
        return None

    # Resolve model
    avail = _available_models()
    if not avail:
        st.session_state["upload_error"] = "No trained model artifacts found. Run `python main.py` first."
        return None

    resolved_name = model_name
    selection_reason = ""
    selection_metric = None
    telco = is_telco_schema(df)
    # Any previous generic pipeline belongs to a different upload.
    st.session_state["pipeline_result"] = None

    if telco:
        if resolved_name is None or resolved_name == "Auto (Best)":
            try:
                sel = select_best_model(df)
                resolved_name = sel["name"]
                selection_reason = sel["reason"]
                selection_metric = sel.get("metric_value")
            except Exception as e:
                resolved_name = next(iter(avail))
                selection_reason = f"Auto-selection failed ({e}); using first available artifact."
                selection_metric = None
        else:
            # Look up saved metric for display
            from src.model_registry import get_saved_metrics
            saved = get_saved_metrics(resolved_name)
            if saved:
                selection_metric = saved.get("roc_auc")
                selection_reason = (
                    f"Manually selected by user. "
                    f"Saved validation ROC-AUC: {selection_metric:.4f} (F1={saved.get('f1', 0):.4f})."
                )
            else:
                selection_reason = f"Manually selected by user ('{resolved_name}')."

    # Run predictions
    try:
        if telco:
            model = load_model_by_name(resolved_name)
            predicted = predict_batch(df, model=model, require_schema=True)
        else:
            # Non-Telco: need target column for generic pipeline
            target_candidates = [
                c for c in df.columns
                if any(k in c.lower() for k in ["churn", "target", "outcome", "attrition", "exited"])
            ]
            if target_candidates:
                target_col = target_candidates[0]
                id_col = _find_col(df, ["customerid", "customer_id", "custid", "accountid", "id"],
                                   default=None, exact=True)
                pipeline = run_generic_pipeline(df, target_col, id_col)
                predicted = predict_generic_batch(df, pipeline, target_col, id_col)
                st.session_state["pipeline_result"] = pipeline
                st.session_state["uploaded_target"] = target_col
                # For non-telco, note the generic model actually used
                best_row = next(
                    (r for r in pipeline["results"] if r.get("model") == pipeline["best_name"]),
                    pipeline["results"][0] if pipeline["results"] else {},
                )
                resolved_name = f"Generic ({pipeline['best_name']})"
                selection_reason = (
                    f"Non-Telco schema detected. Trained a schema-specific generic pipeline. "
                    f"Best model in pipeline: {pipeline['best_name']} "
                    f"(ROC-AUC={best_row.get('roc_auc', 0):.4f}."
                )
                selection_metric = best_row.get("roc_auc")
            else:
                raise ValueError(
                    "Dataset does not match Telco schema and has no churn/target column. "
                    "Add a binary churn column or upload a Telco-compatible dataset."
                )
    except Exception as exc:
        st.session_state["upload_error"] = str(exc)
        st.session_state["uploaded_predictions"] = None
        return None

    # Persist to SQLite
    try:
        db_path, run_id = store_predictions_to_sqlite(predicted)
        st.session_state["latest_run_id"] = run_id
    except Exception as exc:
        print(f"SQLite Sync Note: {exc}")

    st.session_state["upload_error"] = None
    st.session_state["uploaded_predictions"] = predicted
    st.session_state["active_model_name"] = resolved_name
    st.session_state["active_model_selection_reason"] = selection_reason
    st.session_state["active_model_metric_value"] = selection_metric

    # Also remember which df hash produced these predictions (for cache invalidation)
    st.session_state["_pred_df_id"] = id(df)
    return predicted


def _ensure_predictions():
    """Re-run predictions if the active df exists but no predictions yet."""
    if (st.session_state.get("uploaded_predictions") is None
            and st.session_state.get("uploaded_df") is not None):
        _run_and_store_predictions(
            st.session_state["uploaded_df"],
            model_name=st.session_state.get("active_model_name"),
        )


def _active_metric_rows():
    """
    Canonical per-model metrics for the dataset currently loaded.

    Returns ``(rows, note, kind)`` where ``kind`` is ``"generic"`` when the
    active upload is non-Telco (metrics come from the pipeline trained on that
    upload) or ``"saved"`` for the benchmark Telco model results. Every page
    that shows per-model scores must call this so the numbers always agree.
    """
    active_df = st.session_state.get("uploaded_df")
    pipe = st.session_state.get("pipeline_result")
    use_generic = bool(pipe) and active_df is not None and not is_telco_schema(active_df)
    if use_generic:
        return canonical_metric_rows(pipeline_result=pipe, use_pipeline=True)
    return canonical_metric_rows(available=_available_models())


def _money(value, available: bool, decimals: int = 0) -> str:
    """Format a currency figure, or 'N/A' when the dataset has no money column."""
    if not available:
        return "N/A"
    return f"${float(value):,.{decimals}f}"


def _rev_titles(rev: dict) -> tuple[str, str, str, str]:
    """
    Return ``(at_risk_title, at_risk_note, second_title, second_note)`` adapted to
    the detected money column so a stock value (balance) is never labelled MRR.
    """
    if not rev.get("available"):
        return (
            "Revenue at Risk", "No monetary column in dataset",
            "Annualized Exposure", "Unavailable for this dataset",
        )
    label = rev.get("label") or "Revenue"
    if rev.get("is_monthly"):
        return (
            "Monthly Revenue at Risk", f"{label} held by churners",
            "Annualized ARR Exposure", "12-month projected risk",
        )
    return (
        f"{label} at Risk", "Value held by predicted churners",
        f"Average {label} (Churners)", "Mean value of a predicted churner",
    )


def _load_benchmark():
    """Load the benchmark Telco dataset and run predictions."""
    demo_path = os.path.join(ROOT, "data", "raw", "telco_churn_raw.csv")
    if os.path.exists(demo_path):
        df = pd.read_csv(demo_path)
        st.session_state["uploaded_df"] = df
        st.session_state["uploaded_predictions"] = None
        st.session_state["pipeline_result"] = None
        st.session_state["uploaded_target"] = None
        st.session_state["current_uploaded_file_id"] = "benchmark"
        _run_and_store_predictions(df)
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════
_defaults = {
    "uploaded_df":                   None,
    "uploaded_predictions":          None,
    "uploaded_target":               None,
    "pipeline_result":               None,
    "latest_run_id":                 None,
    "test_results":                  None,
    "sim_epochs":                    None,
    "current_uploaded_file_id":      None,
    "real_epoch_history":            None,
    "real_epoch_total_time":         None,
    "upload_error":                  None,
    "active_model_name":             None,
    "active_model_selection_reason": None,
    "active_model_metric_value":     None,
    "mce_uploaded_df":               None,
    "mce_file_id":                   None,
    "mce_eval_results":              None,
    "mce_compare_results":           None,
    "_pred_df_id":                   None,
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# Auto-load benchmark on very first visit
if st.session_state["uploaded_df"] is None:
    _load_benchmark()

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style="padding:0.4rem 0 0.8rem;">
        <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.35rem;">
            <div style="font-size:1.35rem;color:{T['gold']};">⚜</div>
            <div style="font-family:'Noto Serif',Georgia,serif;font-size:1.15rem;font-weight:700;color:{T['text']} !important;letter-spacing:-0.02em;">
                Business Analytics
            </div>
        </div>
        <div style="font-size:0.75rem;color:{T['gold_deep']} !important;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.2rem;">
            AI Churn Intelligence
        </div>
        <div style="font-size:0.6875rem;color:{T['text_muted']} !important;letter-spacing:0.02em;">
            Alabaster Sovereign Edition
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "Navigation Menu",
        [
            "Dataset Import & Model Selection",
            "Executive Overview",
            "Business Revenue & Financial Risk",
            "Model Comparison & Evaluation",
            "Model Testing & Evaluation",
            "Customer Retention Strategy",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    # Dataset / model status indicator
    if st.session_state["uploaded_df"] is not None:
        n = len(st.session_state["uploaded_df"])
        active_model = st.session_state.get("active_model_name") or "—"
        st.markdown(f"""<div style="background:{T['card']};border:1px solid {T['border']};
            border-radius:8px;padding:0.8rem 1rem;font-size:0.82rem;margin-bottom:0.7rem;box-shadow:0 1px 2px rgba(30,41,59,0.03);">
            <div style="color:{T['green']} !important;font-weight:700;margin-bottom:0.25rem;">● Active Portfolio Loaded</div>
            <div style="color:{T['text_sec']} !important;font-weight:600;">{n:,} customer accounts</div>
            <div style="color:{T['gold_deep']} !important;font-size:0.75rem;margin-top:0.2rem;">🤖 Model: {escape(active_model)}</div>
        </div>""", unsafe_allow_html=True)
        if st.button("🔄 Unload / Reset Data", use_container_width=True):
            for _k in list(_defaults.keys()):
                st.session_state[_k] = _defaults[_k]
            st.rerun()
    else:
        st.markdown(f"""<div style="background:{T['card']};border:1px solid {T['border']};
            border-radius:8px;padding:0.8rem 1rem;font-size:0.82rem;margin-bottom:0.7rem;box-shadow:0 1px 2px rgba(30,41,59,0.03);">
            <div style="color:{T['amber']} !important;font-weight:700;margin-bottom:0.25rem;">○ No Active Dataset</div>
            <div style="color:{T['text_muted']} !important;">Load benchmark or upload custom file</div>
        </div>""", unsafe_allow_html=True)
        if st.button("⚡ Load Benchmark Data (7,043)", type="primary", use_container_width=True):
            with st.spinner("Loading benchmark…"):
                _load_benchmark()
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DATASET IMPORT & MODEL SELECTION  (NEW — first page)
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Dataset Import & Model Selection":
    page_header(
        "Data & Model Hub",
        "Dataset Import & Model Selection",
        "Load or upload a customer dataset, inspect it, then choose which model scores your portfolio."
    )

    # ── Section A: Dataset upload ─────────────────────────────────────────────
    section_heading("1. Load Dataset")

    col_up, col_bench = st.columns([3, 1])
    with col_bench:
        if st.button("⚡ Load Benchmark (7,043 Accounts)", type="primary", use_container_width=True,
                     help="IBM Telco Churn benchmark — all 5 analytics pages are instantly populated."):
            with st.spinner("Loading benchmark dataset and scoring…"):
                _load_benchmark()
            st.rerun()

    with col_up:
        upload = st.file_uploader(
            "Upload CSV or Excel",
            type=["csv", "xlsx", "xls"],
            key="import_uploader",
            label_visibility="collapsed",
            help="Drag and drop a CSV or Excel file. Up to 200 K rows recommended.",
        )

    if upload is not None:
        upload_id = f"{upload.name}_{upload.size}"
        if st.session_state.get("current_uploaded_file_id") != upload_id:
            try:
                df_new = _read_uploaded_file(upload)
                st.session_state["current_uploaded_file_id"] = upload_id
                st.session_state["uploaded_df"] = df_new
                st.session_state["uploaded_predictions"] = None
                st.session_state["active_model_name"] = None
                # A new upload invalidates the previous generic pipeline and any
                # live evaluation results; otherwise stale metrics leak onto the
                # model cards and comparison tables.
                st.session_state["pipeline_result"] = None
                st.session_state["uploaded_target"] = None
                st.session_state["test_results"] = None
                st.session_state["mce_eval_results"] = None
                st.session_state["mce_compare_results"] = None
                st.rerun()
            except Exception as exc:
                st.error(f"Could not read file: {exc}")

    df_active = st.session_state.get("uploaded_df")

    if df_active is None:
        info_box("📁 <strong>No dataset loaded.</strong> Upload a CSV/Excel file above or click <em>Load Benchmark</em>.")
        st.stop()

    # ── Section B: Dataset validation card ───────────────────────────────────
    section_heading("2. Dataset Overview & Validation")
    vinfo = validate_dataset(df_active)

    st.markdown(f"""<div style="background:{T['card']};border:1px solid {T['border']};
        border-radius:8px;padding:0.9rem 1.4rem;display:flex;gap:2.5rem;flex-wrap:wrap;align-items:center;
        margin:0.5rem 0 1rem;box-shadow:0 1px 3px rgba(30,41,59,0.03);">
        <div>
            <span style="color:{T['text_muted']} !important;font-size:0.72rem;text-transform:uppercase;font-weight:700;letter-spacing:0.08em;">Records</span>
            <div style="color:{T['text']} !important;font-family:'Noto Serif',serif;font-size:1.25rem;font-weight:700;font-variant-numeric:tabular-nums;">{vinfo['n_rows']:,}</div>
        </div>
        <div>
            <span style="color:{T['text_muted']} !important;font-size:0.72rem;text-transform:uppercase;font-weight:700;letter-spacing:0.08em;">Features</span>
            <div style="color:{T['text']} !important;font-family:'Noto Serif',serif;font-size:1.25rem;font-weight:700;font-variant-numeric:tabular-nums;">{vinfo['n_cols']}</div>
        </div>
        <div>
            <span style="color:{T['text_muted']} !important;font-size:0.72rem;text-transform:uppercase;font-weight:700;letter-spacing:0.08em;">Missing Values</span>
            <div style="color:{T['amber'] if vinfo['n_missing'] > 0 else T['green']} !important;font-size:1rem;font-weight:700;">{vinfo['n_missing']:,}</div>
        </div>
        <div>
            <span style="color:{T['text_muted']} !important;font-size:0.72rem;text-transform:uppercase;font-weight:700;letter-spacing:0.08em;">Duplicates</span>
            <div style="color:{T['amber'] if vinfo['n_duplicates'] > 0 else T['green']} !important;font-size:1rem;font-weight:700;">{vinfo['n_duplicates']:,}</div>
        </div>
        <div>
            <span style="color:{T['text_muted']} !important;font-size:0.72rem;text-transform:uppercase;font-weight:700;letter-spacing:0.08em;">Ground-Truth Labels</span>
            <div style="color:{T['green'] if vinfo['has_target'] else T['amber']} !important;font-size:1rem;font-weight:700;">{'● ' + escape(vinfo['target_col']) if vinfo['has_target'] else '○ Not Detected'}</div>
        </div>
        <div>
            <span style="color:{T['text_muted']} !important;font-size:0.72rem;text-transform:uppercase;font-weight:700;letter-spacing:0.08em;">Schema</span>
            <div style="color:{T['green'] if vinfo['is_telco'] else T['amber']} !important;font-size:1rem;font-weight:700;">{'● Telco' if vinfo['is_telco'] else '○ Generic'}</div>
        </div>
    </div>""", unsafe_allow_html=True)

    if vinfo["warnings"]:
        for w in vinfo["warnings"]:
            st.warning(w)

    st.caption(vinfo["schema_note"])

    with st.expander("Dataset Preview (first 10 rows)", expanded=False):
        st.dataframe(df_active.head(10), use_container_width=True, hide_index=True)

    # ── Section C: Dynamic model selection ───────────────────────────────────
    st.markdown("---")
    section_heading("3. Model Selection")

    avail = _available_models()
    if not avail:
        st.error("No trained model artifacts found. Run `python main.py` first.")
        st.stop()

    saved_results = load_saved_model_results()
    saved_map = {r["model"]: r for r in saved_results}

    # Auto-select recommendation
    try:
        auto_sel = select_best_model(df_active)
        auto_name = auto_sel["name"]
        auto_reason = auto_sel["reason"]
        auto_auc = auto_sel.get("metric_value")
    except Exception as e:
        auto_name = next(iter(avail))
        auto_reason = f"Auto-selection error: {e}"
        auto_auc = None

    # Current active model (may differ if user changed it)
    current_name = st.session_state.get("active_model_name") or auto_name

    # Model options: "Auto (Best)" plus individual models
    model_options = ["Auto (Best)"] + list(avail.keys())

    # Determine index
    if current_name in model_options:
        default_idx = model_options.index(current_name)
    elif current_name and current_name.startswith("Generic"):
        default_idx = 0
    else:
        default_idx = 0

    sel_col1, sel_col2 = st.columns([1, 2])
    with sel_col1:
        chosen_option = st.selectbox(
            "Choose model",
            model_options,
            index=default_idx,
            key="import_model_selector",
            label_visibility="collapsed",
        )
    with sel_col2:
        desc_key = auto_name if chosen_option == "Auto (Best)" else chosen_option
        desc_text = MODEL_DESCRIPTIONS.get(desc_key, "Trained model artifact.")
        if chosen_option == "Auto (Best)":
            desc_text = f"Auto-selects: <strong>{escape(auto_name)}</strong> — {escape(desc_text)}"
        st.markdown(f"""<div style="background:{T['surface_sub']};border:1px solid {T['border_sub']};
            border-radius:8px;padding:0.6rem 1rem;font-size:0.85rem;margin-top:0.1rem;">
            {desc_text}
        </div>""", unsafe_allow_html=True)

    # Resolve the actual model name to apply
    apply_name = auto_name if chosen_option == "Auto (Best)" else chosen_option

    # ── Per-model score cards ────────────────────────────────────────────────
    # Source of truth is the SHARED canonical table (also used by Model
    # Comparison and Model Testing). Accuracy is the headline number because it
    # is the metric a business reader expects; F1 and AUC are shown beneath.
    metric_rows_active, metric_note_active, metric_kind_active = _active_metric_rows()
    if metric_kind_active == "saved":
        metric_rows_active = [r for r in metric_rows_active if r["Model"] in avail]

    if metric_rows_active:
        heading_txt = (
            "Live Pipeline Performance (held-out split of this dataset)"
            if metric_kind_active == "generic"
            else "Saved Validation Performance (held-out test set)"
        )
        st.markdown(
            f"<div style='font-size:0.82rem;font-weight:700;color:{T['text_muted']};"
            f"margin:0.8rem 0 0.4rem;text-transform:uppercase;letter-spacing:0.05em;'>"
            f"{heading_txt}</div>",
            unsafe_allow_html=True,
        )
        active_name_now = str(st.session_state.get("active_model_name") or "")
        perf_cols = st.columns(len(metric_rows_active))
        for ci, r in enumerate(metric_rows_active):
            is_chosen = (
                r["Model"] == apply_name
                or active_name_now.endswith(f"({r['Model']})")
                or active_name_now == r["Model"]
            )
            border_color = T["gold"] if is_chosen else T["border"]
            name_color = T["gold_deep"] if is_chosen else T["text"]
            # Build the SELECTED badge separately to avoid single-quote HTML conflicts inside f-string
            selected_badge = (
                f'<div style="color:{T["green"]} !important;font-size:0.72rem;'
                f'font-weight:800;margin-top:0.2rem;">&#10003; SELECTED</div>'
            ) if is_chosen else ""
            with perf_cols[ci]:
                st.markdown(
                    f'<div style="background:{T["card"]};border:2px solid {border_color};'
                    f'border-radius:8px;padding:0.65rem 0.8rem;text-align:center;font-size:0.8rem;'
                    f'box-shadow:0 1px 3px rgba(30,41,59,0.04);">'
                    f'<div style="font-weight:800;color:{name_color} !important;margin-bottom:0.3rem;font-size:0.83rem;">{escape(r["Model"])}</div>'
                    f'<div style="color:{T["text"]} !important;font-size:1.05rem;font-weight:800;">Acc {r["Accuracy"]*100:.1f}%</div>'
                    f'<div style="color:{T["text_muted"]} !important;font-size:0.75rem;">F1 {r["F1-Score"]*100:.1f}% | AUC {r["ROC-AUC"]*100:.1f}%</div>'
                    f'{selected_badge}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        st.caption(metric_note_active)

    # Selection rationale
    if metric_kind_active == "generic":
        rationale = (
            f"Non-Telco schema detected, so a schema-specific pipeline was trained "
            f"on this upload and the best of {len(metric_rows_active)} candidate "
            f"models was applied to the whole file."
        )
    elif chosen_option == "Auto (Best)":
        rationale = auto_reason
    elif apply_name in saved_map:
        rationale = (
            f"Manually selected '{apply_name}'. "
            f"Saved validation ROC-AUC: {saved_map[apply_name]['roc_auc']:.4f}, "
            f"F1: {saved_map[apply_name]['f1']:.4f} (held-out test set)."
        )
    else:
        rationale = f"Manually selected '{apply_name}'."
    st.markdown(f"""<div style="background:{T['surface_sub']};border:1px solid {T['border_sub']};
        border-left:4px solid {T['gold']};border-radius:6px;padding:0.6rem 1rem;
        font-size:0.84rem;margin:0.8rem 0 1rem;">
        <strong style="color:{T['gold_deep']} !important;">ℹ Selection Rationale:</strong>
        <span style="color:{T['text_muted']} !important;"> {escape(str(rationale))}</span>
    </div>""", unsafe_allow_html=True)

    # ── Apply button ──────────────────────────────────────────────────────────
    needs_rescore = (
        st.session_state.get("uploaded_predictions") is None
        or (st.session_state.get("active_model_name") != apply_name
            and not (apply_name.startswith("Generic") and
                     str(st.session_state.get("active_model_name", "")).startswith("Generic")))
    )

    if needs_rescore:
        btn_label = f"⚡ Apply & Score Portfolio with {apply_name}"
    else:
        btn_label = f"🔄 Re-Score Portfolio with {apply_name}"

    if st.button(btn_label, type="primary", use_container_width=True):
        with st.spinner(f"Scoring {len(df_active):,} records with {apply_name}…"):
            _run_and_store_predictions(df_active, model_name=apply_name)
        st.rerun()

    # Auto-score on first load
    if st.session_state.get("uploaded_predictions") is None:
        with st.spinner(f"Auto-scoring with {apply_name}…"):
            _run_and_store_predictions(df_active, model_name=apply_name)
        st.rerun()

    # ── Active state banner ───────────────────────────────────────────────────
    predicted = st.session_state.get("uploaded_predictions")
    err = st.session_state.get("upload_error")
    if err:
        st.error(f"Scoring failed: {err}")
    elif predicted is not None:
        active_name = st.session_state.get("active_model_name", apply_name)
        churn_n = int((predicted["Churn Prediction"] == "Will Churn").sum())
        total_n = len(predicted)
        st.markdown(f"""<div style="background:{T['card']};border:1px solid {T['gold']};
            border-left:5px solid {T['gold']};border-radius:8px;padding:1rem 1.4rem;margin-top:1rem;">
            <div style="display:flex;gap:3rem;flex-wrap:wrap;align-items:center;">
                <div>
                    <div style="font-size:0.72rem;text-transform:uppercase;font-weight:700;color:{T['text_muted']} !important;letter-spacing:0.08em;">Active Dataset</div>
                    <div style="font-size:1.1rem;font-weight:800;color:{T['text']} !important;">{total_n:,} accounts scored</div>
                </div>
                <div>
                    <div style="font-size:0.72rem;text-transform:uppercase;font-weight:700;color:{T['text_muted']} !important;letter-spacing:0.08em;">Active Model</div>
                    <div style="font-size:1.1rem;font-weight:800;color:{T['gold_deep']} !important;">🤖 {escape(active_name)}</div>
                </div>
                <div>
                    <div style="font-size:0.72rem;text-transform:uppercase;font-weight:700;color:{T['text_muted']} !important;letter-spacing:0.08em;">Predicted Churn</div>
                    <div style="font-size:1.1rem;font-weight:800;color:{T['red']} !important;">{churn_n:,} ({churn_n/total_n*100:.1f}%)</div>
                </div>
                <div style="flex:1;text-align:right;">
                    <div style="color:{T['green']} !important;font-weight:700;">✓ Portfolio Ready — Navigate to any page →</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EXECUTIVE OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Executive Overview":
    page_header(
        "Executive Telemetry",
        "Executive Overview",
        "High-level portfolio churn risk intelligence, revenue exposure telemetry, and strategic KPIs."
    )

    _ensure_predictions()
    predicted = st.session_state.get("uploaded_predictions")

    if predicted is None:
        info_box("📊 <strong>No predictions available.</strong> Navigate to <em>Dataset Import &amp; Model Selection</em> to load data.")
        st.stop()

    active_model = st.session_state.get("active_model_name", "—")

    # Active model banner
    st.markdown(f"""<div style="background:{T['surface_sub']};border:1px solid {T['border']};
        border-left:4px solid {T['gold']};border-radius:6px;padding:0.5rem 1rem;margin-bottom:1rem;font-size:0.86rem;">
        <span style="color:{T['gold_deep']} !important;font-weight:700;">🤖 Active Model:</span>
        <span style="color:{T['text']} !important;font-weight:700;"> {escape(active_model)}</span>
        <span style="color:{T['text_muted']} !important;font-size:0.78rem;"> — Predictions below are derived from this model on the active dataset.</span>
    </div>""", unsafe_allow_html=True)

    target = st.session_state.get("uploaded_target")
    total = len(predicted)
    churn_count = int((predicted["Churn Prediction"] == "Will Churn").sum())
    churn_rate  = (churn_count / total * 100) if total > 0 else 0
    high_risk   = int(predicted["Risk Level"].str.contains("High", case=False, na=False).sum())

    run_id = st.session_state.get("latest_run_id")
    sql_results = run_analytics(run_id=run_id) if run_id else {}
    rev = get_revenue_metrics(predicted, sql_results)

    # KPI Row
    at_risk_title, at_risk_note, second_title, second_note = _rev_titles(rev)
    if rev["available"]:
        second_value = (_money(rev["annual"], True) if rev["is_monthly"]
                        else _money(rev["churner_arpu"], True))
    else:
        second_value = "N/A"
    cols = st.columns(5)
    kpi_card(cols[0], "👥", "Total Customers",        f"{total:,}",            "Portfolio accounts analyzed",   T["gold_deep"])
    kpi_card(cols[1], "⚠️", "Predicted Churn Rate",  f"{churn_rate:.1f}%",    f"{churn_count:,} accounts at risk", T["red"])
    kpi_card(cols[2], "🔴", "High-Risk Accounts",     f"{high_risk:,}",        "Immediate intervention required", T["red"])
    kpi_card(cols[3], "💰", at_risk_title,            _money(rev["at_risk"], rev["available"]), at_risk_note, T["amber"])
    kpi_card(cols[4], "📈", second_title,             second_value,            second_note,                     T["amber"])

    # Distinguish predicted vs actual churn rate if labels exist
    if target and target in predicted.columns:
        actual_raw = predicted[target].astype(str).str.strip().str.lower()
        actual_n = int(actual_raw.apply(lambda x: 1 if x in {"1","yes","true","churn","will churn","churned","1.0"} else 0).sum())
        st.caption(
            f"ℹ **Actual churn (ground-truth labels):** {actual_n:,} "
            f"({actual_n/total*100:.1f}%) &nbsp;|&nbsp; "
            f"**Predicted churn (model output):** {churn_count:,} ({churn_rate:.1f}%)"
        )

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        section_heading("Portfolio Risk Distribution")
        risk_clean = (predicted["Risk Level"].str.replace("🔴 ", "", regex=False)
                      .str.replace("🟡 ", "", regex=False).str.replace("🟢 ", "", regex=False))
        risk_counts = risk_clean.value_counts().rename_axis("Risk").reset_index(name="Customers")
        fig = px.pie(risk_counts, names="Risk", values="Customers", hole=0.6, color="Risk",
                     color_discrete_map={"High": T["red"], "Medium": T["amber"], "Low": T["green"]})
        fig.update_traces(textinfo="percent+value",
                          textfont=dict(size=12, family="Manrope, sans-serif"),
                          marker=dict(line=dict(color="#FFFFFF", width=2)))
        st.plotly_chart(chart_style(fig, 330), use_container_width=True)

    with c2:
        section_heading("Churn Probability Distribution Deciles")
        probs = predicted["Churn Probability"].copy()
        if probs.max() <= 1.0 and probs.max() > 0:
            probs = probs * 100.0
        bins   = [-0.001, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        labels = ["0-10%","10-20%","20-30%","30-40%","40-50%","50-60%","60-70%","70-80%","80-90%","90-100%"]
        binned = pd.cut(probs, bins=bins, labels=labels)
        bc = binned.value_counts().reindex(labels, fill_value=0).reset_index()
        bc.columns = ["Range", "Count"]
        fig = px.bar(bc, x="Range", y="Count", text="Count", color_discrete_sequence=[T["gold"]])
        fig.update_traces(marker_line_width=0, textposition="outside",
                          textfont=dict(size=11, family="Manrope, sans-serif", color=T["text_sec"]))
        fig.update_layout(xaxis_title="Churn Probability", yaxis_title="Customer Count", bargap=0.25)
        st.plotly_chart(chart_style(fig, 330), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        value_word = "MRR" if rev.get("is_monthly") else "Value"
        section_heading("Revenue at Risk by Contract Type")
        if (sql_results and "churn_by_contract" in sql_results
                and not sql_results["churn_by_contract"].empty
                and sql_results["churn_by_contract"]["Contract"].notna().any()):
            cdf = sql_results["churn_by_contract"][["Contract", "mrr_at_risk"]].copy()
            cdf.columns = ["Contract", f"{value_word} Lost"]
        else:
            mc_col, _, _ = find_revenue_column(predicted)
            cc = find_col(predicted, ["contract", "contract_type"], default=None)
            if mc_col and cc:
                churned = predicted[predicted["Churn Prediction"] == "Will Churn"]
                cdf = (pd.to_numeric(churned[mc_col], errors="coerce").fillna(0)
                       .groupby(churned[cc]).sum().reset_index())
                cdf.columns = ["Contract", f"{value_word} Lost"]
            else:
                cdf = pd.DataFrame()
        if not cdf.empty:
            value_col = cdf.columns[1]
            fig = px.bar(cdf, x="Contract", y=value_col,
                         text=cdf[value_col].map("${:,.0f}".format),
                         color_discrete_sequence=[T["red"]])
            fig.update_traces(marker_line_width=0, textposition="outside",
                              textfont=dict(size=11, family="Manrope, sans-serif", color=T["text_sec"]))
            fig.update_layout(xaxis_title="Contract Type", yaxis_title="Revenue Exposure ($)")
            st.plotly_chart(chart_style(fig, 330), use_container_width=True)
        else:
            st.info("Contract data not available.")

    with c4:
        section_heading("Top Churn Risk Segments")
        segments = risk_segment_summaries(predicted, target)
        if segments:
            for i, seg in enumerate(segments[:6]):
                st.markdown(f"""<div style="background:{T['card']};border:1px solid {T['border']};
                    border-radius:6px;padding:0.65rem 0.95rem;margin-bottom:0.4rem;font-size:0.86rem;box-shadow:0 1px 2px rgba(30,41,59,0.02);">
                    <span style="color:{T['gold_deep']} !important;font-weight:700;font-family:'Noto Serif',serif;margin-right:0.4rem;">{i+1}.</span>
                    <span style="color:{T['text_sec']} !important;font-family:'Manrope',sans-serif;">{escape(seg)}</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("Upload dataset to extract churn segments.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — BUSINESS REVENUE & FINANCIAL RISK
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Business Revenue & Financial Risk":
    page_header(
        "Financial Telemetry",
        "Business Revenue & Financial Risk",
        "Executive-quality revenue impact modeling — value at risk, ARPU benchmarks, and cohort breakdowns."
    )

    _ensure_predictions()
    predicted = st.session_state.get("uploaded_predictions")

    if predicted is None:
        info_box("📊 <strong>No dataset loaded.</strong> Navigate to <em>Dataset Import &amp; Model Selection</em> first.")
        st.stop()

    # SQL Analytics
    run_id = st.session_state.get("latest_run_id")
    if not run_id:
        _, run_id = store_predictions_to_sqlite(predicted)
        st.session_state["latest_run_id"] = run_id
    sql_results = run_analytics(run_id=run_id)

    rev = get_revenue_metrics(predicted, sql_results)
    total_mrr = rev["arpu"] * rev["total"] if rev["total"] > 0 else 1
    mrr_pct   = (rev["at_risk"] / total_mrr * 100) if total_mrr > 0 else 0
    value_word = "MRR" if rev.get("is_monthly") else "Value"
    # Readable noun for notes: "revenue" for a monthly figure, else the detected
    # column's own label (e.g. "account balance") so we never say "portfolio mrr".
    value_word_lower = "revenue" if rev.get("is_monthly") else str(rev.get("label") or "value").lower()
    at_risk_title, at_risk_note, second_title, second_note = _rev_titles(rev)

    # KPI Row
    cols = st.columns(4)
    if rev["available"]:
        second_value = _money(rev["annual"], True) if rev["is_monthly"] else _money(rev["churner_arpu"], True)
        at_risk_note = f"{mrr_pct:.1f}% of portfolio {value_word_lower}"
    else:
        second_value = "N/A"
        at_risk_note = "No monetary column detected"
    kpi_card(cols[0], "💰", at_risk_title,                   _money(rev["at_risk"], rev["available"]), at_risk_note, T["red"])
    kpi_card(cols[1], "📈", second_title,                    second_value,        second_note,                    T["amber"])
    kpi_card(cols[2], "👤", "Average Revenue Per User (ARPU)", _money(rev["arpu"], rev["available"], 2),
             f"Churner ARPU: {_money(rev['churner_arpu'], rev['available'], 2)}", T["gold_deep"])
    kpi_card(cols[3], "🔴", "High-Value Churn Accounts",       f"{rev['high_val']:,}", "Accounts above average spend", T["red"])

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        section_heading("Revenue Loss by Contract Type")
        if ("churn_by_contract" in sql_results and not sql_results["churn_by_contract"].empty
                and sql_results["churn_by_contract"]["Contract"].notna().any()):
            cdf = (sql_results["churn_by_contract"][["Contract", "mrr_at_risk"]]
                   .rename(columns={"mrr_at_risk": f"{value_word} Lost"}))
        else:
            mc_col, _, _ = find_revenue_column(predicted)
            cc = find_col(predicted, ["contract", "contract_type"], None)
            if mc_col and cc and cc in predicted.columns:
                churned = predicted[predicted["Churn Prediction"] == "Will Churn"]
                cdf = (pd.to_numeric(churned[mc_col], errors="coerce").fillna(0)
                       .groupby(churned[cc]).sum().reset_index())
                cdf.columns = ["Contract", f"{value_word} Lost"]
            else:
                cdf = pd.DataFrame()
        if not cdf.empty:
            vcol = cdf.columns[1]
            fig = px.bar(cdf, x="Contract", y=vcol,
                         text=cdf[vcol].map("${:,.0f}".format),
                         color_discrete_sequence=[T["red"]])
            fig.update_traces(marker_line_width=0, textposition="outside",
                              textfont=dict(size=11, family="Manrope, sans-serif", color=T["text_sec"]))
            fig.update_layout(xaxis_title="Contract Type", yaxis_title=f"{value_word} at Risk ($)")
            st.plotly_chart(chart_style(fig, 340), use_container_width=True)
        else:
            st.info("Contract column not available.")

    with c2:
        section_heading("Revenue Exposure by Payment Method")
        if ("churn_by_payment" in sql_results and not sql_results["churn_by_payment"].empty
                and sql_results["churn_by_payment"]["PaymentMethod"].notna().any()):
            pdf = (sql_results["churn_by_payment"][["PaymentMethod", "mrr_at_risk"]]
                   .rename(columns={"PaymentMethod": "Method", "mrr_at_risk": f"{value_word} Lost"}))
        else:
            mc_col, _, _ = find_revenue_column(predicted)
            pc = find_col(predicted, ["paymentmethod", "payment_method"], None)
            if mc_col and pc and pc in predicted.columns:
                churned = predicted[predicted["Churn Prediction"] == "Will Churn"]
                pdf = (pd.to_numeric(churned[mc_col], errors="coerce").fillna(0)
                       .groupby(churned[pc]).sum().reset_index())
                pdf.columns = ["Method", f"{value_word} Lost"]
            else:
                pdf = pd.DataFrame()
        if not pdf.empty:
            vcol = pdf.columns[1]
            fig = px.pie(pdf, names="Method", values=vcol, hole=0.58,
                         color_discrete_sequence=[T["red"],T["amber"],T["gold"],"#775A00"])
            fig.update_traces(marker=dict(line=dict(color="#FFFFFF", width=2)),
                              textfont=dict(family="Manrope, sans-serif", size=11))
            fig.add_annotation(text=f"<b>{_money(rev['at_risk'], rev['available'])}</b><br><span style='font-size:10px'>{value_word} Risk</span>",
                               x=0.5, y=0.5, showarrow=False,
                               font=dict(color=T["text"], size=13, family="Noto Serif, serif"))
            st.plotly_chart(chart_style(fig, 340), use_container_width=True)
        else:
            st.info("Payment method column not available.")

    if "tenure_cohort_analysis" in sql_results and not sql_results["tenure_cohort_analysis"].empty:
        st.markdown("---")
        section_heading("Churn Rate by Tenure Cohort")
        tdf = sql_results["tenure_cohort_analysis"]
        fig = px.bar(tdf, x="tenure_cohort", y="churn_rate_pct",
                     text=tdf["churn_rate_pct"].apply(lambda x: f"{x:.1f}%"),
                     color_discrete_sequence=[T["gold"]])
        fig.update_traces(marker_line_width=0, textposition="outside",
                          textfont=dict(size=11, family="Manrope, sans-serif", color=T["text_sec"]))
        fig.update_layout(xaxis_title="Tenure Cohort", yaxis_title="Churn Rate (%)")
        st.plotly_chart(chart_style(fig, 320), use_container_width=True)

    st.markdown("---")
    section_heading("Executive Financial Impact Summary")
    if rev["available"]:
        if rev["is_monthly"]:
            summary = (
                f"Based on <strong>{rev['total']:,}</strong> portfolio accounts, total monthly revenue at risk is "
                f"<strong>{_money(rev['at_risk'], True, 2)}</strong> ({mrr_pct:.1f}% of portfolio MRR), representing "
                f"an annualized exposure of <strong>{_money(rev['annual'], True, 2)}</strong>. Churner ARPU "
                f"(<strong>{_money(rev['churner_arpu'], True, 2)}</strong>) vs overall ARPU "
                f"(<strong>{_money(rev['arpu'], True, 2)}</strong>). "
                f"<em>Note: annualized figures assume current pricing over a 12-month horizon without retention.</em>"
            )
        else:
            summary = (
                f"Based on <strong>{rev['total']:,}</strong> portfolio accounts, the monetary column detected was "
                f"<strong>{escape(str(rev['label']))}</strong>. Predicted churners hold "
                f"<strong>{_money(rev['at_risk'], True, 2)}</strong> of that value ({mrr_pct:.1f}% of the total "
                f"<strong>{_money(total_mrr, True, 2)}</strong>), averaging "
                f"<strong>{_money(rev['churner_arpu'], True, 2)}</strong> per churner vs "
                f"<strong>{_money(rev['arpu'], True, 2)}</strong> for the portfolio. "
                f"<em>Note: this column is a stock (per-customer) value, so it is not multiplied by 12.</em>"
            )
    else:
        summary = (
            f"Based on <strong>{rev['total']:,}</strong> portfolio accounts, <strong>no monetary column could be "
            f"identified</strong>, so revenue exposure is reported as N/A rather than a fabricated estimate. "
            f"Add a column such as <code>MonthlyCharges</code>, <code>balance</code>, <code>amount</code> or "
            f"<code>ltv</code> to enable revenue-at-risk reporting."
        )
    info_box(summary)

    if "high_risk_customers" in sql_results and not sql_results["high_risk_customers"].empty:
        st.markdown("---")
        section_heading("High-Risk Accounts Drill-Down (SQLite Telemetry)")
        st.caption("Top high-exposure customer accounts queried directly via SQLite parameterized telemetry.")
        st.dataframe(sql_results["high_risk_customers"].copy(), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL COMPARISON & EVALUATION  (enhanced with comparison chart)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Model Comparison & Evaluation":
    page_header(
        "Model Intelligence Lab",
        "Model Comparison & Evaluation",
        "Evaluate, compare, and validate machine learning models for customer churn prediction."
    )

    # ── A. Dataset selector ───────────────────────────────────────────────────
    section_heading("Upload Evaluation Dataset")
    mce_upload = st.file_uploader(
        "Upload CSV dataset for model evaluation",
        type=["csv","xlsx","xls"],
        key="mce_uploader",
        help="Upload a customer dataset to evaluate models. Include a churn/target column for supervised metrics.",
    )

    mce_df = st.session_state.get("mce_uploaded_df")

    if mce_upload is not None:
        mce_file_id = f"{mce_upload.name}_{mce_upload.size}"
        if st.session_state.get("mce_file_id") != mce_file_id:
            try:
                mce_df = _read_uploaded_file(mce_upload)
                st.session_state["mce_uploaded_df"] = mce_df
                st.session_state["mce_file_id"] = mce_file_id
                st.session_state["mce_eval_results"] = None
                st.session_state["mce_compare_results"] = None
                st.session_state["mce_pipeline_result"] = None
            except Exception as exc:
                st.error(f"Could not read file: {exc}")
                mce_df = None
    elif mce_df is None:
        col_bm, col_active = st.columns(2)
        with col_bm:
            if st.button("⚡ Load Benchmark Dataset (7,043)", key="mce_load_bench", use_container_width=True):
                demo_path = os.path.join(ROOT, "data", "raw", "telco_churn_raw.csv")
                if os.path.exists(demo_path):
                    mce_df = pd.read_csv(demo_path)
                    st.session_state["mce_uploaded_df"] = mce_df
                    st.session_state["mce_file_id"] = "benchmark"
                    st.session_state["mce_eval_results"] = None
                    st.session_state["mce_compare_results"] = None
                    st.session_state["mce_pipeline_result"] = None
                    st.rerun()
        with col_active:
            if st.session_state.get("uploaded_df") is not None:
                if st.button("📂 Use Active Portfolio Dataset", key="mce_use_active", use_container_width=True):
                    mce_df = st.session_state["uploaded_df"]
                    st.session_state["mce_uploaded_df"] = mce_df
                    st.session_state["mce_file_id"] = "active"
                    st.session_state["mce_eval_results"] = None
                    st.session_state["mce_compare_results"] = None
                    st.session_state["mce_pipeline_result"] = None
                    st.rerun()

    if mce_df is not None:
        vinfo_mce = validate_dataset(mce_df)
        n_rows, n_cols = vinfo_mce["n_rows"], vinfo_mce["n_cols"]
        n_missing, n_dupes = vinfo_mce["n_missing"], vinfo_mce["n_duplicates"]
        has_target = vinfo_mce["has_target"]
        target_name = vinfo_mce["target_col"]

        st.markdown(f"""<div style="background:{T['card']};border:1px solid {T['border']};
            border-radius:8px;padding:0.9rem 1.4rem;display:flex;gap:2.5rem;flex-wrap:wrap;align-items:center;margin:0.5rem 0 1rem;box-shadow:0 1px 3px rgba(30,41,59,0.03);">
            <div><span style="color:{T['text_muted']} !important;font-size:0.72rem;text-transform:uppercase;font-weight:700;letter-spacing:0.08em;">Records</span>
                 <div style="color:{T['text']} !important;font-family:'Noto Serif',serif;font-size:1.25rem;font-weight:700;">{n_rows:,}</div></div>
            <div><span style="color:{T['text_muted']} !important;font-size:0.72rem;text-transform:uppercase;font-weight:700;letter-spacing:0.08em;">Features</span>
                 <div style="color:{T['text']} !important;font-family:'Noto Serif',serif;font-size:1.25rem;font-weight:700;">{n_cols}</div></div>
            <div><span style="color:{T['text_muted']} !important;font-size:0.72rem;text-transform:uppercase;font-weight:700;letter-spacing:0.08em;">Missing Values</span>
                 <div style="color:{T['amber'] if n_missing>0 else T['green']} !important;font-size:1rem;font-weight:700;">{n_missing:,}</div></div>
            <div><span style="color:{T['text_muted']} !important;font-size:0.72rem;text-transform:uppercase;font-weight:700;letter-spacing:0.08em;">Duplicates</span>
                 <div style="color:{T['amber'] if n_dupes>0 else T['green']} !important;font-size:1rem;font-weight:700;">{n_dupes:,}</div></div>
            <div><span style="color:{T['text_muted']} !important;font-size:0.72rem;text-transform:uppercase;font-weight:700;letter-spacing:0.08em;">Ground-Truth Labels</span>
                 <div style="color:{T['green'] if has_target else T['amber']} !important;font-size:1rem;font-weight:700;">{'● ' + escape(target_name) if has_target else '○ Not Detected'}</div></div>
        </div>""", unsafe_allow_html=True)

        with st.expander("Preview Dataset", expanded=False):
            st.dataframe(mce_df.head(10), use_container_width=True, hide_index=True)

        # ── B. Model selector ─────────────────────────────────────────────────
        st.markdown("---")
        section_heading("Select Model")
        available_model_map = _available_models()
        if not available_model_map:
            st.warning("No trained model artifacts found. Run `python main.py` first.")
            st.stop()

        model_options = list(available_model_map.keys()) + ["🔀 Compare All Models"]
        # Default to the globally active model
        _active_global = st.session_state.get("active_model_name", "")
        _clean_global = _active_global.replace("Generic (", "").rstrip(")") if _active_global else ""
        _default_idx = (model_options.index(_clean_global) if _clean_global in model_options
                        else model_options.index("🔀 Compare All Models"))

        selected_model = st.selectbox(
            "Choose a model to evaluate",
            model_options,
            index=_default_idx,
            key="mce_model_selector",
        )
        compare_mode = selected_model == "🔀 Compare All Models"

        if not compare_mode:
            st.markdown(f"""<div style="background:{T['surface_sub']};border:1px solid {T['border_sub']};
                border-radius:8px;padding:0.8rem 1.2rem;margin:0.5rem 0 1rem;font-size:0.88rem;">
                <strong style="color:{T['gold_deep']} !important;">{escape(selected_model)}</strong>
                <span style="color:{T['text_muted']} !important;"> — {escape(MODEL_DESCRIPTIONS.get(selected_model,'Trained model artifact.'))}</span>
                <br/><span style="color:{T['green']} !important;font-size:0.8rem;font-weight:700;">● Trained artifact available</span>
            </div>""", unsafe_allow_html=True)

        # Evaluation config
        with st.expander("⚙️ Evaluation Configuration", expanded=False):
            cfg_c1, cfg_c2, cfg_c3 = st.columns(3)
            with cfg_c1:
                mce_test_size = st.slider("Test Size", 0.10, 0.40, 0.20, 0.05, key="mce_test_size")
            with cfg_c2:
                mce_seed = st.number_input("Random Seed", value=42, step=1, key="mce_seed")
            with cfg_c3:
                if has_target:
                    all_cols = mce_df.columns.tolist()
                    mce_target_col = st.selectbox(
                        "Target Column", all_cols,
                        index=all_cols.index(target_name) if target_name in all_cols else 0,
                        key="mce_target_col")
                else:
                    mce_target_col = None

        # ── C. Run button ────────────────────────────────────────────────────
        st.markdown("---")
        btn_label = ("🚀 Run Model Evaluation & Predict Churn" if not compare_mode
                     else "🚀 Run All-Model Comparison")
        if st.button(btn_label, type="primary", use_container_width=True, key="mce_run_btn"):
            with st.spinner("Executing model evaluation pipeline…"):
                try:
                    import joblib as _jl

                    telco_eval = is_telco_schema(mce_df)

                    # For a non-Telco upload we train ONE generic pipeline and then
                    # score with each of its candidate models. Previously the
                    # pipeline was rebuilt inside the loop and the pipeline's own
                    # best model was used every time, so every "model" showed the
                    # same metrics.
                    pipe_eval = None
                    eval_models = {}
                    if not telco_eval and has_target and mce_target_col:
                        id_col = _find_col(mce_df, ["customerid","customer_id","id"],
                                           default=None, exact=True)
                        with st.spinner("Training candidate models on this dataset…"):
                            pipe_eval = run_generic_pipeline(mce_df, mce_target_col, id_col)
                        eval_models = dict(pipe_eval.get("all_models") or {})
                        # Keep this evaluation's pipeline separate from the active
                        # portfolio pipeline so the two datasets cannot overwrite
                        # each other's metrics.
                        st.session_state["mce_pipeline_result"] = pipe_eval
                        if not compare_mode:
                            # Honour the selected model when it exists in this
                            # pipeline; otherwise fall back to the pipeline's best.
                            pick = selected_model if selected_model in eval_models else pipe_eval.get("best_name")
                            if pick in eval_models:
                                eval_models = {pick: eval_models[pick]}
                    else:
                        eval_models = {
                            n: _jl.load(os.path.join(MODELS_DIR, f))
                            for n, f in (
                                {selected_model: available_model_map[selected_model]}
                                if not compare_mode else available_model_map
                            ).items()
                        }

                    all_results = []
                    for m_name, model_obj in eval_models.items():
                        # Score
                        if telco_eval:
                            pred_df = predict_batch(mce_df, model=model_obj, require_schema=True)
                        elif pipe_eval is not None:
                            pred_df = predict_generic_batch(
                                mce_df, pipe_eval, mce_target_col, id_col=None, model=model_obj
                            )
                        else:
                            pred_df = predict_batch(mce_df, model=model_obj, require_schema=False)

                        # Metrics
                        metrics = {"model": m_name}
                        if has_target and mce_target_col and mce_target_col in pred_df.columns:
                            ev = evaluate_predictions(pred_df, mce_target_col)
                            metrics.update(ev)
                            metrics["has_truth"] = True
                        else:
                            metrics["has_truth"] = False
                        metrics["pred_df"] = pred_df
                        all_results.append(metrics)

                    if compare_mode:
                        st.session_state["mce_compare_results"] = all_results
                        st.session_state["mce_eval_results"]    = all_results[0] if all_results else None
                    else:
                        st.session_state["mce_eval_results"]    = all_results[0] if all_results else None
                        st.session_state["mce_compare_results"] = None
                    st.rerun()
                except Exception as exc:
                    import traceback
                    st.error(f"Evaluation failed: {exc}")
                    st.code(traceback.format_exc())

        # ── D. Results ───────────────────────────────────────────────────────
        eval_res    = st.session_state.get("mce_eval_results")
        compare_res = st.session_state.get("mce_compare_results")

        if eval_res is not None or compare_res is not None:
            st.markdown("---")
            primary = eval_res if eval_res else (compare_res[0] if compare_res else None)
            if primary is None:
                st.stop()

            pred_df   = primary["pred_df"]
            has_truth = primary.get("has_truth", False)

            # ── Canonical metrics for the model being shown ──────────────────
            # Page 4 must display the same accuracy/F1/AUC numbers as pages 1
            # and 5. Those come from the shared metric table, never from
            # re-scoring this dataset on screen (which mixes splits and
            # thresholds and produced the "81.9% vs 73%" confusion).
            _mce_pipe0 = st.session_state.get("mce_pipeline_result")
            if _mce_pipe0 and not is_telco_schema(mce_df):
                _can_rows = metric_rows(_mce_pipe0.get("results") or [])
                _can_note = (
                    "Metrics are the held-out split results of the generic pipeline trained "
                    "on this dataset — the same values shown on every other page."
                )
            else:
                _can_rows, _can_note, _k = _active_metric_rows()
            canon = {r["Model"]: r for r in _can_rows}
            crow = canon.get(primary.get("model"))

            def _cm(key):
                """Canonical metric value, falling back to the live run."""
                if crow:
                    return crow[key]
                return float(primary.get(key) or 0.0)

            section_heading("Evaluation Results")
            total_eval = len(pred_df)
            churn_n    = int((pred_df["Churn Prediction"] == "Will Churn").sum())
            churn_pct  = churn_n / total_eval * 100 if total_eval > 0 else 0

            if has_truth or crow:
                cols = st.columns(6)
                kpi_card(cols[0], "📊", "Records Processed", f"{total_eval:,}", "Total evaluated",       T["gold_deep"])
                kpi_card(cols[1], "⚠️", "Predicted Churn",   f"{churn_pct:.1f}%", f"{churn_n:,} churners", T["red"])
                kpi_card(cols[2], "🎯", "Accuracy",    f"{_cm('Accuracy')*100:.1f}%",  "Overall correct",     T["green"])
                kpi_card(cols[3], "⚖️", "F1-Score",    f"{_cm('F1-Score')*100:.1f}%",  "Harmonic balance",    T["gold_deep"])
                kpi_card(cols[4], "📈", "ROC-AUC",     f"{_cm('ROC-AUC')*100:.1f}%",   "Discrimination",      T["gold_deep"])
                kpi_card(cols[5], "📏", "Precision",   f"{_cm('Precision')*100:.1f}%", "Positive predictive", T["amber"])
                # Distinguish actual vs predicted churn
                if "n_churn_actual" in primary:
                    st.caption(
                        f"ℹ **Actual churn (ground-truth):** {primary['n_churn_actual']:,} "
                        f"({primary['churn_rate_actual']:.1f}%) — "
                        f"**Predicted churn:** {primary['n_churn_predicted']:,} "
                        f"({primary['churn_rate_predicted']:.1f}%) — "
                        f"Decision threshold: {primary['threshold']:.2f}"
                    )
                st.caption(f"📊 {_can_note}")
                if crow and abs(_cm("Accuracy") - float(primary.get("accuracy") or 0)) > 0.0005:
                    st.caption(
                        "ℹ The supervised metrics above are the canonical held-out values. "
                        "The confusion matrix and ROC curve below come from re-scoring the "
                        "uploaded rows and may therefore differ slightly."
                    )
            else:
                cols = st.columns(4)
                kpi_card(cols[0], "📊", "Records Processed", f"{total_eval:,}", "Total evaluated", T["gold_deep"])
                kpi_card(cols[1], "⚠️", "Predicted Churn",   f"{churn_pct:.1f}%", f"{churn_n:,} churners", T["red"])
                kpi_card(cols[2], "🎯", "Accuracy",  "N/A", "No ground-truth labels",  T["text_muted"])
                kpi_card(cols[3], "📈", "ROC-AUC",   "N/A", "No ground-truth labels",  T["text_muted"])
                info_box("⚠️ <strong>Ground-truth labels not provided.</strong> Add a churn/target column to compute supervised metrics.")

            # Churn distribution + histogram
            st.markdown("---")
            ch1, ch2 = st.columns(2)
            with ch1:
                section_heading("Churn Prediction Distribution")
                _vc = pred_df["Churn Prediction"].value_counts().reset_index()
                _vc.columns = ["Prediction", "Count"]
                fig_pie = px.pie(_vc, names="Prediction", values="Count", hole=0.6,
                                 color="Prediction",
                                 color_discrete_map={"Will Churn":T["red"],"Will Stay":T["green"]})
                fig_pie.update_traces(textinfo="percent+value",
                                      textfont=dict(size=12, family="Manrope, sans-serif"),
                                      marker=dict(line=dict(color="#FFFFFF", width=2)))
                st.plotly_chart(chart_style(fig_pie, 330), use_container_width=True)
            with ch2:
                section_heading("Churn Probability Distribution")
                probs_d = pred_df["Churn Probability"].copy()
                if probs_d.max() <= 1.0 and probs_d.max() > 0:
                    probs_d = probs_d * 100.0
                bins   = [-0.001,10,20,30,40,50,60,70,80,90,100]
                labels = ["0-10%","10-20%","20-30%","30-40%","40-50%","50-60%","60-70%","70-80%","80-90%","90-100%"]
                binned = pd.cut(probs_d, bins=bins, labels=labels)
                bc = binned.value_counts().reindex(labels, fill_value=0).reset_index()
                bc.columns = ["Range","Count"]
                fig_h = px.bar(bc, x="Range", y="Count", text="Count", color_discrete_sequence=[T["gold"]])
                fig_h.update_traces(marker_line_width=0, textposition="outside",
                                    textfont=dict(size=11, family="Manrope, sans-serif", color=T["text_sec"]))
                fig_h.update_layout(bargap=0.25)
                st.plotly_chart(chart_style(fig_h, 330), use_container_width=True)

            # ── Model Performance Comparison grouped bar chart ────────────────
            # Uses the SAME canonical table as the cards on page 1 and the table
            # on page 5. Live evaluation results are only used when the page-4
            # dataset is non-Telco, because in that case the generic pipeline
            # produces genuine holdout metrics for exactly this upload.
            st.markdown("---")
            section_heading("Model Performance Comparison")
            metrics_list = _can_rows
            eval_source_note = _can_note

            if metrics_list:
                mdf = pd.DataFrame(metrics_list)
                mdf_melted = mdf.melt(id_vars="Model", var_name="Metric", value_name="Score")
                model_colors = {
                    "Logistic Regression": T["gold"],
                    "Random Forest":       T["green"],
                    "XGBoost":             "#E2703A",
                    "LightGBM":            "#7C3AED",
                    "SVM":                 T["amber"],
                    "Voting Ensemble":     T["red"],
                }
                fig_comp = px.bar(
                    mdf_melted, x="Metric", y="Score", color="Model", barmode="group",
                    text=mdf_melted["Score"].apply(lambda v: f"{v*100:.1f}%"),
                    color_discrete_map=model_colors,
                )
                fig_comp.update_traces(textposition="outside",
                                       textfont=dict(size=9, family="Manrope, sans-serif"))
                fig_comp.update_layout(yaxis=dict(range=[0, 1.1]), xaxis_title="", yaxis_title="Score (0–1)")
                st.plotly_chart(chart_style(fig_comp, 420), use_container_width=True)
                st.caption(f"📊 {eval_source_note}")

                # Numeric table
                styled = mdf.style.format({c: "{:.2%}" for c in mdf.columns if c != "Model"})
                st.dataframe(styled, use_container_width=True, hide_index=True)

            # Individual model benchmark KPI cards — same canonical table
            st.markdown("---")
            section_heading("Individual Model Benchmarks")
            best_auc_name = max(metrics_list, key=lambda x: x["ROC-AUC"])["Model"] if metrics_list else None
            for row in metrics_list:
                m_name = row["Model"]
                is_best = (m_name == best_auc_name)
                badge = (
                    f'<span style="background:{T["green"]};color:#fff;font-size:0.65rem;'
                    f'padding:2px 8px;border-radius:4px;font-weight:700;margin-left:0.6rem;'
                    f'letter-spacing:0.05em;">BEST ROC-AUC</span>'
                ) if is_best else ""
                st.markdown(
                    f'<div style="margin:1rem 0 0.5rem;">'
                    f'<span style="color:{T["gold_deep"]} !important;font-family:\'Noto Serif\',serif;'
                    f'font-size:1.05rem;font-weight:700;">{escape(m_name)}</span>{badge}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                mc = st.columns(5)
                kpi_card(mc[0], "🎯", "ACCURACY",  f"{row['Accuracy']*100:.1f}%",   "Overall accuracy",         T["green"])
                kpi_card(mc[1], "⚖️", "PRECISION", f"{row['Precision']*100:.1f}%",  "Positive predictive",      T["gold_deep"])
                kpi_card(mc[2], "📈", "RECALL",    f"{row['Recall']*100:.1f}%",     "True positive sensitivity", T["amber"])
                kpi_card(mc[3], "📊", "F1-SCORE",  f"{row['F1-Score']*100:.1f}%",   "Harmonic balance",         T["gold_deep"])
                kpi_card(mc[4], "📏", "ROC-AUC",   f"{row['ROC-AUC']*100:.1f}%",   "Discrimination",           T["green"])

            # Confusion matrix — only shown when an evaluation has been run on this page
            if has_truth:
                st.markdown("---")
                ch3, ch4 = st.columns(2)
                with ch3:
                    section_heading("Confusion Matrix (Evaluation Run)")
                    cm = [[primary["tn"], primary["fp"]], [primary["fn"], primary["tp"]]]
                    fig_cm = go.Figure(data=go.Heatmap(
                        z=cm, x=["Predicted Retained","Predicted Churn"],
                        y=["Actual Retained","Actual Churn"],
                        text=[[str(v) for v in row] for row in cm], texttemplate="%{text}",
                        colorscale=[[0,"#FAFAF7"],[1,T["gold"]]], showscale=False,
                        textfont=dict(size=18, color=T["text"], family="Manrope, sans-serif"),
                    ))
                    fig_cm.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
                    st.plotly_chart(chart_style(fig_cm, 320), use_container_width=True)
                with ch4:
                    section_heading("Classification Breakdown")
                    cc = st.columns(2)
                    kpi_card(cc[0], "✅", "True Positives",  f"{primary['tp']:,}", "Correctly predicted churn",    T["green"])
                    kpi_card(cc[1], "✅", "True Negatives",  f"{primary['tn']:,}", "Correctly predicted retained",  T["green"])
                    cc2 = st.columns(2)
                    kpi_card(cc2[0], "⚠️", "False Positives", f"{primary['fp']:,}", "False alarms",               T["amber"])
                    kpi_card(cc2[1], "❌", "False Negatives", f"{primary['fn']:,}", "Missed churners",             T["red"])

            # ROC curves — only shown when evaluation run produced probabilities
            if has_truth and primary.get("probs") is not None:
                st.markdown("---")
                section_heading("ROC Curves (Evaluation Run)")
                from sklearn.metrics import roc_curve as _roc_curve
                roc_colors = [T["gold"],T["green"],"#E2703A","#7C3AED",T["amber"],T["red"]]
                fig_roc = go.Figure()
                fpr, tpr, _ = _roc_curve(primary["actual_bin"], primary["probs"])
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode="lines",
                    name=f"{primary['model']} (AUC={primary.get('roc_auc',0) or 0:.3f})",
                    line=dict(color=roc_colors[0], width=2.5),
                ))
                fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                              name="Random Baseline",
                                              line=dict(color="#999999", width=1.5, dash="dash")))
                fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                st.plotly_chart(chart_style(fig_roc, 400), use_container_width=True)

            # Feature importance
            st.markdown("---")
            section_heading("Feature Importance")
            fi_shown = False
            if not compare_mode and eval_res:
                try:
                    import joblib as _jl2
                    _m_path = os.path.join(MODELS_DIR, available_model_map[eval_res["model"]])
                    m_obj = _jl2.load(_m_path)
                    if hasattr(m_obj, "feature_importances_"):
                        feat_names = (list(m_obj.feature_names_in_)
                                      if hasattr(m_obj, "feature_names_in_") else
                                      [f"F{i}" for i in range(len(m_obj.feature_importances_))])
                        fi = pd.DataFrame({"Feature": feat_names, "Importance": m_obj.feature_importances_})
                        fi = fi.sort_values("Importance", ascending=False).head(15)
                        fig_fi = px.bar(fi[::-1], x="Importance", y="Feature", orientation="h",
                                        color_discrete_sequence=[T["gold"]],
                                        text=fi[::-1]["Importance"].apply(lambda v: f"{v*100:.1f}%"))
                        fig_fi.update_traces(marker_line_width=0, textposition="outside",
                                             textfont=dict(size=11, family="Manrope, sans-serif", color=T["text_sec"]))
                        fig_fi.update_layout(xaxis_title="Importance", yaxis_title="")
                        st.plotly_chart(chart_style(fig_fi, 420), use_container_width=True)
                        fi_shown = True
                except Exception:
                    pass
            if not fi_shown:
                fi_path = os.path.join(MODELS_DIR, "feature_importance.csv")
                if os.path.exists(fi_path):
                    fi_df = pd.read_csv(fi_path).head(15)
                    fig_fi = px.bar(fi_df[::-1], x="Importance", y="Feature", orientation="h",
                                    color_discrete_sequence=[T["gold"]],
                                    text=fi_df[::-1]["Importance"].apply(lambda v: f"{v*100:.1f}%"))
                    fig_fi.update_traces(marker_line_width=0, textposition="outside",
                                         textfont=dict(size=11, family="Manrope, sans-serif", color=T["text_sec"]))
                    fig_fi.update_layout(xaxis_title="Importance", yaxis_title="")
                    st.plotly_chart(chart_style(fig_fi, 420), use_container_width=True)
                else:
                    st.info("Feature importance not available for this model type.")

            # Customer-level predictions table + downloads
            st.markdown("---")
            section_heading("Customer-Level Predictions")
            id_col_name = _find_col(pred_df, ["customerid","customer_id","id"], default=None, exact=True)
            disp_cols = []
            if id_col_name:
                disp_cols.append(id_col_name)
            if has_truth and target_name and target_name in pred_df.columns:
                disp_cols.append(target_name)
            disp_cols += ["Churn Prediction","Churn Probability","Risk Level"]
            for c in ["tenure","MonthlyCharges","Contract","InternetService"]:
                if c in pred_df.columns and c not in disp_cols:
                    disp_cols.append(c)
            disp_cols = [c for c in disp_cols if c in pred_df.columns]
            st.dataframe(pred_df[disp_cols].head(500), use_container_width=True, hide_index=True, height=400)

            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button("📥 Download Predictions CSV", pred_df.to_csv(index=False).encode(),
                                   "model_evaluation_predictions.csv", "text/csv", use_container_width=True)
            with dl2:
                if compare_res and any(r.get("has_truth") for r in compare_res):
                    rpt = pd.DataFrame([{
                        "Model": r["model"],
                        "Accuracy": r.get("accuracy","N/A"),
                        "Precision": r.get("precision","N/A"),
                        "Recall": r.get("recall","N/A"),
                        "F1": r.get("f1","N/A"),
                        "ROC-AUC": r.get("roc_auc","N/A"),
                    } for r in compare_res if r.get("has_truth")])
                    st.download_button("📥 Download Comparison Report CSV", rpt.to_csv(index=False).encode(),
                                       "model_comparison_report.csv", "text/csv", use_container_width=True)

            if st.button("🔄 Clear Results", key="mce_clear_results", use_container_width=True):
                st.session_state["mce_eval_results"] = None
                st.session_state["mce_compare_results"] = None
                st.rerun()
    else:
        info_box("📂 <strong>No dataset loaded.</strong> Upload a CSV, load the benchmark, or use the active portfolio above.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — MODEL TESTING & EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Model Testing & Evaluation":
    page_header(
        "Model Performance",
        "Model Testing & Evaluation",
        "Comprehensive evaluation against ground-truth datasets, benchmark audits, and optimization convergence."
    )

    # ── Section 1: Canonical per-model metrics ───────────────────────────────
    section_heading("Trained Model Performance (Production Benchmarks)")
    _p5_rows, _p5_note, _p5_kind = _active_metric_rows()
    if _p5_rows:
        # Same criterion as training: highest ROC-AUC, F1 breaks ties.
        _p5_best = max(_p5_rows, key=lambda r: (r["ROC-AUC"], r["F1-Score"]))["Model"]
        for mr in _p5_rows:
            name = mr["Model"]
            is_best = (name == _p5_best)
            badge = (' <span class="status-pill status-gold" style="margin-left:0.5rem;">PRODUCTION</span>'
                     if is_best else "")
            st.markdown(f"<h4 style='margin-top:1rem;'>{escape(name)}{badge}</h4>", unsafe_allow_html=True)
            cols = st.columns(5)
            kpi_card(cols[0], "🎯", "Accuracy",   f"{mr['Accuracy']*100:.1f}%",   "Overall accuracy")
            kpi_card(cols[1], "📏", "Precision",  f"{mr['Precision']*100:.1f}%",  "Positive predictive value")
            kpi_card(cols[2], "🔍", "Recall",     f"{mr['Recall']*100:.1f}%",     "True positive sensitivity", T["green"])
            kpi_card(cols[3], "⚖️", "F1-Score",   f"{mr['F1-Score']*100:.1f}%",   "Harmonic balance")
            kpi_card(cols[4], "📈", "ROC-AUC",    f"{mr['ROC-AUC']*100:.1f}%",    "Discrimination capability", T["gold_deep"])
        st.caption(f"📊 {_p5_note}")
    else:
        st.info("No saved model results found. Run `python main.py` to train benchmark models.")

    # ── Feature importance ────────────────────────────────────────────────────
    fi_path = os.path.join(MODELS_DIR, "feature_importance.csv")
    if os.path.exists(fi_path):
        st.markdown("---")
        section_heading("Feature Importance (Production Model)")
        st.caption("Relative predictive weight of top features.")
        fi_df = pd.read_csv(fi_path).head(10)
        fig_fi = px.bar(fi_df[::-1], x="Importance", y="Feature", orientation="h",
                        color_discrete_sequence=[T["gold"]],
                        text=fi_df[::-1]["Importance"].apply(lambda v: f"{v*100:.1f}%"))
        fig_fi.update_traces(marker_line_width=0, textposition="outside",
                             textfont=dict(size=11, family="Manrope, sans-serif", color=T["text_sec"]))
        fig_fi.update_layout(xaxis_title="Relative Feature Importance Weight", yaxis_title="")
        st.plotly_chart(chart_style(fig_fi, 340), use_container_width=True)

    st.markdown("---")

    # ── Section 2: Live Ground-Truth Evaluation ───────────────────────────────
    section_heading("Live Ground-Truth Test Dataset Evaluation")
    st.caption("Upload a labeled test dataset to compute live accuracy, confusion matrix, and actual-vs-predicted comparison.")

    # Model selector for this section — defaults to the globally active model
    avail_te = _available_models()
    te_model_options = list(avail_te.keys())
    _active_g = st.session_state.get("active_model_name", "")
    _clean_g  = _active_g.replace("Generic (","").rstrip(")") if _active_g else ""
    _te_default = te_model_options.index(_clean_g) if _clean_g in te_model_options else 0

    te_col1, te_col2 = st.columns([1, 2])
    with te_col1:
        te_model_sel = st.selectbox(
            "Model for Live Evaluation",
            te_model_options,
            index=_te_default,
            key="te_model_selector",
            help="Select the model to evaluate against the uploaded test dataset.",
        )
    with te_col2:
        if te_model_sel != _clean_g and _clean_g:
            st.markdown(f"""<div style="background:{T['amber_bg']};border:1px solid {T['amber_border']};
                border-radius:6px;padding:0.45rem 0.9rem;font-size:0.84rem;margin-top:0.15rem;">
                ⚠️ <strong>Override:</strong> You selected <strong>{escape(te_model_sel)}</strong>
                (globally active: <strong>{escape(_clean_g)}</strong>).
                Results below apply to <strong>{escape(te_model_sel)}</strong> only.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div style="background:{T['green_bg']};border:1px solid {T['green_border']};
                border-radius:6px;padding:0.45rem 0.9rem;font-size:0.84rem;margin-top:0.15rem;">
                ✓ <strong>{escape(te_model_sel)}</strong> matches the globally active model.
                Results will be consistent with the Executive Overview.
            </div>""", unsafe_allow_html=True)

    test_upload = st.file_uploader(
        "Upload Test Dataset (with Churn column)",
        type=["csv","xlsx","xls"],
        key="uploader_test",
    )

    test_df = None
    if test_upload is not None:
        try:
            test_df = _read_uploaded_file(test_upload)
            st.success(f"Loaded ground-truth dataset: {len(test_df):,} rows × {len(test_df.columns)} columns")
        except Exception as exc:
            st.error(f"Could not read file: {exc}")
    elif st.session_state["uploaded_df"] is not None:
        test_df = st.session_state["uploaded_df"]
        st.info(f"Using active portfolio dataset: {len(test_df):,} accounts — "
                f"NOTE: This is the same dataset used for portfolio scoring; "
                "for an unbiased test, upload a separate labeled dataset.")

    if test_df is not None:
        cols_list = test_df.columns.tolist()
        target_col = find_col(test_df, ["churn","target","actual_churn","outcome"], default=cols_list[-1])

        c1, c2 = st.columns(2)
        with c1:
            selected_target = st.selectbox(
                "Ground-Truth Actual Churn Column", cols_list,
                index=cols_list.index(target_col) if target_col in cols_list else len(cols_list)-1,
            )
        with c2:
            id_col = find_col(test_df, ["customerid","id","name"], default=None)
            selected_id = st.selectbox(
                "Customer ID Column (Optional)",
                ["None"] + cols_list,
                index=(cols_list.index(id_col)+1) if id_col and id_col in cols_list else 0,
            )

        if st.button("⚡ Execute Model Evaluation Against Ground Truth", type="primary", use_container_width=True):
            with st.spinner(f"Evaluating {te_model_sel} against ground-truth labels…"):
                try:
                    id_col_name = None if selected_id == "None" else selected_id
                    te_model_obj = load_model_by_name(te_model_sel)

                    # Score — use the production model pipeline for Telco schema;
                    # only fall back to generic pipeline for genuinely non-Telco data.
                    if is_telco_schema(test_df):
                        eval_pred = predict_batch(test_df, model=te_model_obj, require_schema=True)
                    else:
                        pipeline = run_generic_pipeline(test_df, selected_target, id_col_name)
                        # Score with the SELECTED model (falling back to the
                        # pipeline's best when the name has no generic analogue)
                        # instead of always using the pipeline's own best model.
                        gen_models = pipeline.get("all_models") or {}
                        chosen_gen = gen_models.get(te_model_sel)
                        eval_pred = predict_generic_batch(
                            test_df, pipeline, selected_target, id_col_name,
                            model=chosen_gen if chosen_gen is not None else pipeline["best_model"],
                        )

                    # Evaluate using the shared function (single source of truth for TP/TN/FP/FN)
                    ev = evaluate_predictions(eval_pred, selected_target)

                    eval_pred["Actual"]    = np.where(ev["actual_bin"] == 1, "Churn", "Retained")
                    eval_pred["Predicted"] = np.where(ev["pred_bin"]   == 1, "Churn", "Retained")
                    eval_pred["Match"]     = np.where(ev["actual_bin"] == ev["pred_bin"], "✅ Correct", "❌ Missed")

                    st.session_state["test_results"] = {
                        **ev,
                        "df":         eval_pred,
                        "model_name": te_model_sel,
                        "dataset_id": "uploaded_test" if test_upload else "active_portfolio",
                    }
                    st.rerun()
                except Exception as exc:
                    import traceback
                    st.error(f"Evaluation failed: {exc}")
                    st.code(traceback.format_exc())

        if st.session_state.get("test_results"):
            res = st.session_state["test_results"]
            res_model = res.get("model_name", te_model_sel)
            res_dataset = res.get("dataset_id", "unknown")

            # Identify which dataset these results belong to
            is_portfolio = (res_dataset == "active_portfolio")
            is_same_as_active = (res_model == _clean_g or res_model == st.session_state.get("active_model_name",""))

            if is_portfolio and is_same_as_active:
                info_box(
                    "ℹ <strong>Consistency check:</strong> These results use the same model and dataset as the Executive Overview. "
                    "Predicted churn count and probabilities will match the portfolio view exactly."
                )
            elif not is_same_as_active:
                st.warning(
                    f"Results below are for **{res_model}** (override), not the globally active "
                    f"**{_clean_g}**. They will differ from the Executive Overview."
                )

            acc_pct  = res["accuracy"] * 100
            prec_pct = res["precision"] * 100
            rec_pct  = res["recall"] * 100
            f1_pct   = res["f1"] * 100

            cols = st.columns(4)
            kpi_card(cols[0], "🎯", "Test Accuracy",
                     f"{acc_pct:.1f}%",
                     f"{res['tp']+res['tn']:,} / {res['n_total']:,} correct", T["green"])
            kpi_card(cols[1], "📏", "Precision",  f"{prec_pct:.1f}%", "Flagged churners correct",  T["gold_deep"])
            kpi_card(cols[2], "🔍", "Recall",     f"{rec_pct:.1f}%",  "Actual churners captured",  T["green"])
            kpi_card(cols[3], "⚖️", "F1-Score",   f"{f1_pct:.1f}%",   "Harmonic balance",           T["gold_deep"])

            st.caption(
                f"Model: **{res_model}** | "
                f"Actual churn: **{res['n_churn_actual']:,}** ({res['churn_rate_actual']:.1f}%) | "
                f"Predicted churn: **{res['n_churn_predicted']:,}** ({res['churn_rate_predicted']:.1f}%) | "
                f"Threshold: **{res['threshold']:.2f}**"
            )

            # Reconcile with the production benchmark shown at the top of this page
            _bench = next((r for r in _p5_rows if r["Model"] == res_model), None)
            if _bench:
                delta = (res["accuracy"] - _bench["Accuracy"]) * 100
                st.caption(
                    f"📊 **Production benchmark accuracy for {res_model}: "
                    f"{_bench['Accuracy']*100:.1f}%** (held-out split during training). "
                    f"Live accuracy above is {res['accuracy']*100:.1f}% on the rows you just uploaded "
                    f"({delta:+.1f} pp). A gap is expected because this is a different set of rows; "
                    f"the benchmark figure is the canonical value used on every other page."
                )

            # Confusion matrix
            st.markdown("---")
            section_heading("Confusion Matrix & Classification Breakdown")
            c1, c2 = st.columns(2)
            with c1:
                cm = [[res["tn"], res["fp"]], [res["fn"], res["tp"]]]
                fig = go.Figure(data=go.Heatmap(
                    z=cm, x=["Predicted Retained","Predicted Churn"],
                    y=["Actual Retained","Actual Churn"],
                    text=[[str(v) for v in row] for row in cm], texttemplate="%{text}",
                    colorscale=[[0,"#FAFAF7"],[1,T["gold"]]], showscale=False,
                    textfont=dict(size=18, color=T["text"], family="Manrope, sans-serif"),
                ))
                fig.update_layout(xaxis_title="Predicted Class", yaxis_title="Actual Ground Truth")
                st.plotly_chart(chart_style(fig, 320), use_container_width=True)
            with c2:
                cols2 = st.columns(2)
                kpi_card(cols2[0], "✅", "True Positives",  f"{res['tp']:,}", "Correctly predicted churn",    T["green"])
                kpi_card(cols2[1], "✅", "True Negatives",  f"{res['tn']:,}", "Correctly predicted retained",  T["green"])
                cols3 = st.columns(2)
                kpi_card(cols3[0], "⚠️", "False Positives", f"{res['fp']:,}", "False alarms",                T["amber"])
                kpi_card(cols3[1], "❌", "False Negatives", f"{res['fn']:,}", "Missed churners",              T["red"])

            section_heading("Actual vs. Predicted Comparison Ledger")
            disp = [c for c in res["df"].columns
                    if c in ["CustomerID","Name","Contract","tenure","MonthlyCharges",
                             "Actual","Predicted","Churn Probability","Match"]]
            if not disp:
                disp = res["df"].columns.tolist()[:6] + ["Actual","Predicted","Match"]
            disp = [c for c in disp if c in res["df"].columns]
            st.dataframe(res["df"][disp].head(500), use_container_width=True, hide_index=True, height=380)

    # ── Section 3: Neural-net convergence telemetry ───────────────────────────
    st.markdown("---")
    section_heading("Model Training & Validation Convergence Telemetry")
    st.caption("Execute real iterative neural network optimization. Convergence loss, accuracy, and per-epoch timestamps are computed from the active dataset.")

    ctl_col1, ctl_col2 = st.columns([2, 1])
    with ctl_col1:
        user_epochs = st.slider("Epoch Horizon", min_value=5, max_value=50, value=20, step=1)
    with ctl_col2:
        st.markdown("<div style='height:1.75rem;'></div>", unsafe_allow_html=True)
        btn_start_epochs = st.button("⚡ Start Real Epoch Training", type="primary", use_container_width=True)

    if btn_start_epochs:
        src_df = st.session_state.get("uploaded_df")
        if src_df is None:
            demo_path = os.path.join(ROOT, "data", "raw", "telco_churn_raw.csv")
            if os.path.exists(demo_path):
                src_df = pd.read_csv(demo_path)
        if src_df is None or src_df.empty:
            st.error("No dataset available for epoch training.")
        else:
            with st.spinner(f"Running {user_epochs} epochs…"):
                try:
                    t_col = find_col(src_df, ["churn","target","outcome","attrition"], default=src_df.columns[-1])
                    y_raw = src_df[t_col].astype(str).str.lower().str.strip()
                    y     = y_raw.apply(lambda x: 1 if x in ["1","yes","true","churn","will churn"] else 0).values
                    id_c  = find_col(src_df, ["customerid","customer_id","id","name"], default=None)
                    drop_c = [c for c in [t_col, id_c] if c and c in src_df.columns]
                    X_raw = src_df.drop(columns=drop_c)
                    X_enc = pd.get_dummies(X_raw, drop_first=True).apply(pd.to_numeric, errors="coerce").fillna(0)
                    scaler_ep = StandardScaler()
                    X_scaled  = scaler_ep.fit_transform(X_enc)
                    strat = y if len(np.unique(y)) > 1 else None
                    X_tr, X_val, y_tr, y_val = train_test_split(
                        X_scaled, y, test_size=0.20, random_state=42, stratify=strat)
                    mlp = MLPClassifier(hidden_layer_sizes=(32,16), activation="relu",
                                        solver="adam", learning_rate_init=0.01,
                                        warm_start=True, max_iter=1, random_state=42)
                    history = []
                    prog_bar = st.progress(0.0)
                    status_slot = st.empty()
                    total_t0 = time.time()
                    for ep in range(1, user_epochs + 1):
                        ep_t0 = time.time()
                        mlp.fit(X_tr, y_tr)
                        ep_dur = time.time() - ep_t0
                        tr_acc  = accuracy_score(y_tr,  mlp.predict(X_tr))  * 100.0
                        tr_loss = log_loss(y_tr,  mlp.predict_proba(X_tr))
                        val_acc = accuracy_score(y_val, mlp.predict(X_val)) * 100.0
                        val_loss= log_loss(y_val, mlp.predict_proba(X_val))
                        history.append({
                            "Epoch": ep,
                            "Timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                            "Epoch Time (s)": round(ep_dur, 3),
                            "Train Accuracy (%)": round(tr_acc, 2),
                            "Val Accuracy (%)":   round(val_acc, 2),
                            "Train Loss":  round(tr_loss, 4),
                            "Val Loss":    round(val_loss, 4),
                        })
                        prog_bar.progress(ep / user_epochs)
                        status_slot.markdown(
                            f"**Epoch `{ep}/{user_epochs}`:** Train Acc: **{tr_acc:.1f}%** | "
                            f"Val Acc: **{val_acc:.1f}%** | Loss: **{tr_loss:.4f}** ({ep_dur:.3f}s)")
                    prog_bar.empty(); status_slot.empty()
                    st.session_state["real_epoch_history"]    = pd.DataFrame(history)
                    st.session_state["real_epoch_total_time"] = round(time.time() - total_t0, 3)
                    st.success(f"Completed {user_epochs} epochs in {st.session_state['real_epoch_total_time']}s!")
                except Exception as exc:
                    st.error(f"Epoch training error: {exc}")

    if st.session_state.get("real_epoch_history") is not None:
        ep_df   = st.session_state["real_epoch_history"]
        last    = ep_df.iloc[-1]
        tot_t   = st.session_state.get("real_epoch_total_time", 0.0)
        k_cols  = st.columns(5)
        kpi_card(k_cols[0], "🔄", "Epochs Completed",   f"{len(ep_df)}",                   f"Finished in {tot_t}s",         T["gold_deep"])
        kpi_card(k_cols[1], "🎯", "Final Train Acc",     f"{last['Train Accuracy (%)']:.1f}%", "Training set fit",           T["green"])
        kpi_card(k_cols[2], "⚖️", "Final Val Acc",       f"{last['Val Accuracy (%)']:.1f}%",   "Generalization accuracy",    T["green"])
        kpi_card(k_cols[3], "📉", "Final Train Loss",    f"{last['Train Loss']:.4f}",           "Log loss minimization",      T["gold_deep"])
        kpi_card(k_cols[4], "🔍", "Final Val Loss",      f"{last['Val Loss']:.4f}",             "Holdout cross-entropy",      T["red"])

        curv1, curv2 = st.columns(2)
        with curv1:
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(x=ep_df["Epoch"], y=ep_df["Train Accuracy (%)"], mode="lines+markers",
                                          name="Train Accuracy (%)",
                                          line=dict(color=T["gold"], width=2.5),
                                          marker=dict(size=5, color=T["gold"])))
            fig_acc.add_trace(go.Scatter(x=ep_df["Epoch"], y=ep_df["Val Accuracy (%)"], mode="lines+markers",
                                          name="Validation Accuracy (%)",
                                          line=dict(color=T["text_sec"], width=2, dash="dash"),
                                          marker=dict(size=5, color=T["text_sec"])))
            fig_acc.update_layout(xaxis_title="Epoch", yaxis_title="Accuracy (%)",
                                   title=dict(text="Train vs. Validation Accuracy",
                                              font=dict(family="Noto Serif, serif", size=14)))
            st.plotly_chart(chart_style(fig_acc, 320), use_container_width=True)
        with curv2:
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(x=ep_df["Epoch"], y=ep_df["Train Loss"], mode="lines+markers",
                                           name="Train Loss",
                                           line=dict(color=T["gold"], width=2.5),
                                           marker=dict(size=5, color=T["gold"])))
            fig_loss.add_trace(go.Scatter(x=ep_df["Epoch"], y=ep_df["Val Loss"], mode="lines+markers",
                                           name="Validation Loss",
                                           line=dict(color=T["red"], width=2, dash="dash"),
                                           marker=dict(size=5, color=T["red"])))
            fig_loss.update_layout(xaxis_title="Epoch", yaxis_title="Cross-Entropy / Log Loss",
                                    title=dict(text="Train vs. Validation Loss",
                                               font=dict(family="Noto Serif, serif", size=14)))
            st.plotly_chart(chart_style(fig_loss, 320), use_container_width=True)

        st.markdown("---")
        section_heading("Real Epoch Execution Telemetry Ledger")
        st.dataframe(ep_df, use_container_width=True, hide_index=True, height=280)
        st.download_button("📥 Export Epoch Telemetry (CSV)", ep_df.to_csv(index=False).encode(),
                           "real_epoch_training_telemetry.csv", "text/csv")
    else:
        info_box("⚡ <strong>Ready:</strong> Select epoch horizon above and click <strong>'Start Real Epoch Training'</strong>.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — CUSTOMER RETENTION STRATEGY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Customer Retention Strategy":
    page_header(
        "Strategic Action",
        "Customer Retention Strategy",
        "Actionable retention strategies and prescriptive playbooks prioritized by customer value, contract exposure, and behavioral attrition signals."
    )

    _ensure_predictions()
    predicted = st.session_state.get("uploaded_predictions")

    if predicted is None:
        info_box("📊 <strong>No dataset loaded.</strong> Navigate to <em>Dataset Import &amp; Model Selection</em> first.")
        st.stop()

    total       = len(predicted)
    churn_count = int((predicted["Churn Prediction"] == "Will Churn").sum())

    risk_clean = (predicted["Risk Level"].astype(str)
                  .str.replace("🔴 ", "", regex=False)
                  .str.replace("🟡 ", "", regex=False)
                  .str.replace("🟢 ", "", regex=False).str.strip())
    predicted_copy = predicted.copy()
    predicted_copy["_risk_clean"] = risk_clean

    high_risk   = predicted_copy[predicted_copy["_risk_clean"].str.lower() == "high"]
    medium_risk = predicted_copy[predicted_copy["_risk_clean"].str.lower() == "medium"]

    # Detect the money column the same way pages 2/3 do, so a bank-style
    # dataset reports its balance/amount instead of a fake $0 (or a made-up $70).
    rev6        = get_revenue_metrics(predicted)
    mc, rev6_is_monthly, rev6_label = find_revenue_column(predicted)
    rev6_word   = (rev6_label or "Value")
    tenure_col  = find_col(predicted, ["tenure","months"], None)
    contract_col= find_col(predicted, ["contract","contract_type"], None)
    payment_col = find_col(predicted, ["paymentmethod","payment_method"], None)
    internet_col= find_col(predicted, ["internetservice","internet_service"], None)
    sec_col     = find_col(predicted, ["onlinesecurity","online_security"], None)
    sup_col     = find_col(predicted, ["techsupport","tech_support"], None)

    def _val(v) -> str:
        """Currency figure, or N/A when the dataset has no monetary column."""
        return _money(v, rev6["available"])

    def _flow(v) -> str:
        """'$X/mo ($Y/yr ARR)' for monthly columns, '$X of value' for stock columns."""
        if not rev6["available"]:
            return "N/A"
        if rev6_is_monthly:
            return f"{_money(v, True)}/mo ({_money(float(v) * 12, True)}/yr ARR)"
        return f"{_money(v, True)} of {str(rev6_word).lower()} value"

    _per_period = "mo" if rev6_is_monthly else "account"
    # Tile wording: monthly columns are revenue; stock columns keep their own name
    # ("Account Balance") so a bank file never sees the word "MRR".
    rev6_tile   = "Revenue" if rev6_is_monthly else (rev6_label or "Value")

    cols = st.columns(4)
    kpi_card(cols[0], "🔴", "High-Risk Accounts",      f"{len(high_risk):,}",  "Immediate intervention required", T["red"])
    kpi_card(cols[1], "🟡", "Medium-Risk Accounts",    f"{len(medium_risk):,}","Proactive nurture recommended",   T["amber"])
    if mc:
        high_value = pd.to_numeric(high_risk[mc], errors="coerce").fillna(0.0).sum()
        kpi_card(cols[2], "💰", f"High-Risk {rev6_tile}",
                 _val(high_value), "Value at immediate risk", T["red"])
    else:
        kpi_card(cols[2], "💰", "High-Risk Value at Risk", "N/A",
                 "No monetary column detected", T["red"])
    ret_target = len(high_risk) + len(medium_risk)
    kpi_card(cols[3], "🎯", "Retention Target Accounts", f"{ret_target:,}",
             f"{ret_target/total*100:.1f}% of portfolio", T["gold_deep"])

    st.markdown("---")

    # ── ROI Simulator ─────────────────────────────────────────────────────────
    section_heading("Retention Campaign ROI & Capital Recovery Simulator")
    st.caption("Simulate proactive retention campaigns and forecast the customer value preserved.")
    sim_col1, sim_col2 = st.columns([1, 1])
    with sim_col1:
        st.markdown(f"<div style='font-size:0.88rem;font-weight:700;color:{T['text']};margin-bottom:0.5rem;'>Campaign Parameters</div>", unsafe_allow_html=True)
        sim_succ_rate    = st.slider("Intervention Success Rate (%)", 10, 60, 25, 5)
        sim_cost_per_cust= st.slider("Incentive Cost per Customer ($)", 5, 60, 15, 5)

    sim_pool          = high_risk if len(high_risk) > 0 else (medium_risk if len(medium_risk) > 0 else predicted_copy)
    sim_target_count  = len(sim_pool)
    sim_retained      = int(sim_target_count * sim_succ_rate / 100)
    avg_mc            = pd.to_numeric(sim_pool[mc], errors="coerce").mean() if mc else 0.0
    if pd.isna(avg_mc):
        avg_mc = 0.0
    sim_value_saved   = sim_retained * avg_mc
    sim_annual_saved  = sim_value_saved * 12.0 if rev6_is_monthly else sim_value_saved
    sim_cost          = sim_target_count * sim_cost_per_cust
    sim_net           = sim_annual_saved - sim_cost
    sim_roi           = (sim_net / sim_cost * 100) if sim_cost > 0 else 0

    with sim_col2:
        st.markdown(f"<div style='font-size:0.88rem;font-weight:700;color:{T['text']};margin-bottom:0.5rem;'>Projected Returns</div>", unsafe_allow_html=True)
        rc1, rc2 = st.columns(2)
        kpi_card(rc1, "🛡️", "Customers Saved",  f"{sim_retained:,}",      f"{sim_succ_rate}% of {sim_target_count:,}",    T["green"])
        kpi_card(rc2, "💵", f"Preserved {rev6_tile}",
                 _val(sim_value_saved) if rev6["available"] else "N/A",
                 f"Per {'month' if rev6_is_monthly else 'account'} value", T["green"])
        rc3, rc4 = st.columns(2)
        kpi_card(rc3, "📈", "Annualized Value Protected" if rev6_is_monthly else "Total Value Protected",
                 _val(sim_annual_saved) if rev6["available"] else "N/A",
                 "12-month capital protected" if rev6_is_monthly else "One-period value", T["gold_deep"])
        kpi_card(rc4, "💎", "Net Campaign ROI", f"{sim_roi:,.0f}%",
                 f"{_val(sim_net) if rev6['available'] else 'N/A'} net gain", T["green"])

    st.caption("⚠ Projections assume current pricing and a constant success rate. Actual outcomes depend on execution and subscriber responsiveness.")

    st.markdown("---")

    # ── Priority Outreach Queue ───────────────────────────────────────────────
    section_heading("Highest-Risk Customers — Priority Outreach Queue")
    st.caption("Targeted account queue with algorithmically prescribed retention interventions.")

    outreach_pool = high_risk.copy() if len(high_risk) > 0 else (
        medium_risk.copy() if len(medium_risk) > 0 else predicted_copy.copy())

    avg_portfolio_spend = pd.to_numeric(predicted_copy[mc], errors="coerce").mean() if mc else 0.0
    if pd.isna(avg_portfolio_spend):
        avg_portfolio_spend = 0.0

    def assign_action_protocol(row):
        c_val = str(row.get(contract_col, "")).lower() if contract_col else ""
        t_val = pd.to_numeric(row.get(tenure_col, 99), errors="coerce") if tenure_col else 99
        m_val = pd.to_numeric(row.get(mc, 0), errors="coerce") if mc else 0
        p_val = pd.to_numeric(row.get("Churn Probability", 0), errors="coerce")
        s_val = str(row.get(sec_col, "")).lower() if sec_col else ""
        if "month" in c_val:
            return "📋 12-Month Renewal Incentive (15% Off 3 Mos)"
        elif m_val >= avg_portfolio_spend:
            return "💎 VIP Concierge & Speed Upgrade"
        elif t_val <= 12:
            return "🆕 High-Touch Onboarding & Tutorial Cadence"
        elif "no" in s_val:
            return "🛡️ Free 90-Day Security & Support Suite"
        elif p_val >= 70.0:
            return "🚨 Urgent Retention Specialist Intervention"
        else:
            return "📞 Proactive Satisfaction Check-In & Voucher"

    outreach_pool["Strategic Intervention"] = outreach_pool.apply(assign_action_protocol, axis=1)

    q_filter = st.radio(
        "Filter Outreach Queue",
        ["All Priority Accounts","Month-to-Month Contracts Only",
         "High-Value VIPs (Above Average Spend)","Early Lifecycle (Tenure ≤ 12 Mo)"],
        horizontal=True,
    )
    filtered_queue = outreach_pool.copy()
    if q_filter == "Month-to-Month Contracts Only" and contract_col and contract_col in filtered_queue.columns:
        filtered_queue = filtered_queue[filtered_queue[contract_col].astype(str).str.lower().str.contains("month", na=False)]
    elif q_filter == "High-Value VIPs (Above Average Spend)" and mc and mc in filtered_queue.columns:
        filtered_queue = filtered_queue[pd.to_numeric(filtered_queue[mc], errors="coerce") >= avg_portfolio_spend]
    elif q_filter == "Early Lifecycle (Tenure ≤ 12 Mo)" and tenure_col and tenure_col in filtered_queue.columns:
        filtered_queue = filtered_queue[pd.to_numeric(filtered_queue[tenure_col], errors="coerce") <= 12]

    id_col_q   = find_col(predicted, ["customerid","customer_id","id"], None)
    show_cols  = []
    if id_col_q and id_col_q in filtered_queue.columns: show_cols.append(id_col_q)
    if tenure_col and tenure_col in filtered_queue.columns: show_cols.append(tenure_col)
    if contract_col and contract_col in filtered_queue.columns: show_cols.append(contract_col)
    if mc and mc in filtered_queue.columns: show_cols.append(mc)
    show_cols += ["Churn Probability","Risk Level","Strategic Intervention"]
    show_cols   = [c for c in show_cols if c in filtered_queue.columns]

    sorted_q = filtered_queue.sort_values("Churn Probability", ascending=False)
    queue_view = sorted_q[show_cols].head(50)
    if mc and mc in queue_view.columns:
        queue_view = queue_view.rename(columns={mc: f"{rev6_word} ({_per_period})"})
    st.dataframe(queue_view, use_container_width=True, hide_index=True, height=360)
    st.download_button(
        "📥 Export Priority Outreach Queue (CSV for CRM)",
        sorted_q[show_cols].to_csv(index=False).encode(),
        "priority_churn_outreach_queue.csv", "text/csv",
    )

    st.markdown("---")

    # ── Diagnosed Attrition Problems ─────────────────────────────────────────
    section_heading("Diagnosed Attrition Problems & Strategic Solutions")
    st.caption("Strategic diagnosis dynamically synthesized from the active dataset.")

    total_subs     = len(predicted_copy)
    churn_mask     = ((predicted_copy["_risk_clean"].str.lower() == "high") |
                      (predicted_copy["Churn Prediction"].astype(str).str.lower().str.contains("churn", na=False)))
    portfolio_churn= (churn_mask.mean() * 100) if total_subs > 0 else 0.0
    portfolio_mrr  = pd.to_numeric(predicted_copy[mc], errors="coerce").sum() if mc else 0.0
    avg_spend      = pd.to_numeric(predicted_copy[mc], errors="coerce").mean() if mc else 0.0
    if pd.isna(avg_spend):
        avg_spend = 0.0

    problems_data = []

    if contract_col and contract_col in predicted_copy.columns:
        c_stats = predicted_copy.groupby(contract_col).agg(
            count=("Churn Prediction","count"),
            rate=("Churn Prediction", lambda s: (s=="Will Churn").mean()*100)
        ).sort_values("rate", ascending=False)
        worst_c = str(c_stats.index[0])
        worst_c_rate = float(c_stats.iloc[0]["rate"])
        best_c  = str(c_stats.index[-1])
        best_c_rate  = float(c_stats.iloc[-1]["rate"])
        c_risk_mult  = worst_c_rate / max(best_c_rate, 0.2)
        c_pool = predicted_copy[(predicted_copy[contract_col].astype(str)==worst_c) &
                                 (predicted_copy["_risk_clean"].str.lower().isin(["high","medium"]))]
        p1_mrr = pd.to_numeric(c_pool[mc], errors="coerce").sum() if mc else 0.0
        problems_data.append({
            "id":"PROB-01","category":"Contract & Switching Barriers","icon":"⚠️",
            "priority":"CRITICAL RISK" if worst_c_rate>=25 else "HIGH RISK","color":T["red"],
            "title":f"Problem 1: {worst_c} Contract Volatility ({c_risk_mult:.1f}x Higher Attrition)",
            "problem":f"Subscribers on '{worst_c}' contracts exhibit a {worst_c_rate:.1f}% churn rate (vs {best_c_rate:.1f}% for '{best_c}', a {c_risk_mult:.1f}x risk ratio). This cohort exposes {len(c_pool):,} accounts and {_flow(p1_mrr)} to immediate defection.",
            "root_cause":f"'{worst_c}' contracts provide zero switching barrier, enabling easy defection at any billing cycle.",
            "solution":f"Targeted '{worst_c}' to Annual Upgrade Program: 15–20% billing discount on the first 3 cycles plus a complimentary bandwidth tier upgrade.",
            "action":f"Dispatch automated renewal workflows within 48 hours to all {len(c_pool):,} '{worst_c}' at-risk accounts.",
            "target":f"{len(c_pool):,} accounts on '{worst_c}' with elevated churn risk",
            "impact":f"Directly insulates {_flow(p1_mrr)}.",
            "sla":"Within 48 Hours",
        })

    if mc and mc in predicted_copy.columns:
        vip_pool = predicted_copy[(pd.to_numeric(predicted_copy[mc], errors="coerce") >= avg_spend) &
                                   (predicted_copy["_risk_clean"].str.lower().isin(["high","medium"]))]
        vip_mrr  = pd.to_numeric(vip_pool[mc], errors="coerce").sum()
        vip_avg  = pd.to_numeric(vip_pool[mc], errors="coerce").mean() if len(vip_pool)>0 else avg_spend
        vip_rev_share = (vip_mrr / portfolio_mrr * 100) if portfolio_mrr > 0 else 0.0
        vip_rate = (vip_pool["Churn Prediction"]=="Will Churn").mean()*100 if len(vip_pool)>0 else portfolio_churn
        problems_data.append({
            "id":"PROB-02","category":"VIP & Margin Protection","icon":"💎",
            "priority":"CRITICAL RISK","color":T["red"],
            "title":f"Problem 2: High-Value VIP Account Attrition ({_val(vip_mrr)} at Risk)",
            "problem":f"{len(vip_pool):,} accounts spending above portfolio average ({_val(vip_avg)}) show a {vip_rate:.1f}% churn rate, representing {vip_rev_share:.1f}% of total {str(rev6_word).lower()}.",
            "root_cause":"High-spend subscribers demand enterprise-grade QoS and dedicated account care.",
            "solution":f"White-Glove VIP Concierge & Loyalty Shield for the {len(vip_pool):,} high-spend accounts.",
            "action":f"Same-day outreach for all {len(vip_pool):,} VIP accounts (SLA < 12h) with priority network routing.",
            "target":f"{len(vip_pool):,} accounts spending >= {_val(avg_spend)}",
            "impact":f"Protects {_flow(vip_mrr)}.",
            "sla":"Same-Day Executive Outreach (SLA < 12h)",
        })

    if tenure_col and tenure_col in predicted_copy.columns:
        tenure_s  = pd.to_numeric(predicted_copy[tenure_col], errors="coerce")
        early_mask= tenure_s <= 12
        early_pool= predicted_copy[early_mask & (predicted_copy["_risk_clean"].str.lower().isin(["high","medium"]))]
        early_rate= (predicted_copy[early_mask]["Churn Prediction"]=="Will Churn").mean()*100 if early_mask.sum()>0 else portfolio_churn
        early_mrr = pd.to_numeric(early_pool[mc], errors="coerce").sum() if mc else 0.0
        problems_data.append({
            "id":"PROB-03","category":"Lifecycle & Onboarding Friction","icon":"🆕",
            "priority":"HIGH RISK","color":T["amber"],
            "title":f"Problem 3: Early-Lifecycle First-Year Drop-Off ({early_rate:.1f}% Churn Spike)",
            "problem":f"{len(early_pool):,} subscribers in their first 12 months exhibit a {early_rate:.1f}% churn rate, jeopardizing {_flow(early_mrr)} in unamortized CAC.",
            "root_cause":"Installation friction and first-month billing adjustments produce buyer remorse before habits form.",
            "solution":f"High-Touch First-Year Onboarding at Days 14, 30, and 60 for {len(early_pool):,} first-year accounts.",
            "action":f"Trigger automated onboarding sequence with proactive health checks for all {len(early_pool):,} early-stage accounts.",
            "target":f"{len(early_pool):,} subscribers in first 12 months",
            "impact":f"Insulates {_flow(early_mrr)}.",
            "sla":"Days 14, 30 & 60 Post-Activation",
        })

    if payment_col and payment_col in predicted_copy.columns:
        p_stats = predicted_copy.groupby(payment_col).agg(
            rate=("Churn Prediction", lambda s: (s=="Will Churn").mean()*100)
        ).sort_values("rate", ascending=False)
        worst_p = str(p_stats.index[0])
        worst_p_rate = float(p_stats.iloc[0]["rate"])
        best_p  = str(p_stats.index[-1])
        best_p_rate  = float(p_stats.iloc[-1]["rate"])
        pay_pool= predicted_copy[(predicted_copy[payment_col].astype(str)==worst_p) &
                                  (predicted_copy["_risk_clean"].str.lower().isin(["high","medium"]))]
        pay_mrr = pd.to_numeric(pay_pool[mc], errors="coerce").sum() if mc else 0.0
        problems_data.append({
            "id":"PROB-04","category":"Product Stickiness & Billing Friction","icon":"💳",
            "priority":"HIGH RISK" if worst_p_rate>=20 else "MEDIUM RISK","color":T["amber"],
            "title":f"Problem 4: {worst_p} Billing Friction ({worst_p_rate:.1f}% Churn Rate)",
            "problem":f"'{worst_p}' shows the highest billing friction cohort at {worst_p_rate:.1f}% churn (vs {best_p_rate:.1f}% for '{best_p}'). {len(pay_pool):,} accounts and {_flow(pay_mrr)} at risk.",
            "root_cause":"Manual payment methods create recurring cancellation evaluation moments.",
            "solution":f"Frictionless Automated Payment migration incentive — $5.00/mo discount for switching from '{worst_p}'.",
            "action":f"Dispatch billing notification to {len(pay_pool):,} '{worst_p}' subscribers before their next billing cycle.",
            "target":f"{len(pay_pool):,} subscribers paying via '{worst_p}'",
            "impact":f"Stabilizes {_flow(pay_mrr)}.",
            "sla":"Next Billing Cycle",
        })

    if sec_col and sec_col in predicted_copy.columns:
        no_sec_mask = predicted_copy[sec_col].astype(str).str.lower().isin(["no","0","false"])
        yes_sec_mask= predicted_copy[sec_col].astype(str).str.lower().isin(["yes","1","true"])
        no_sec_rate = (predicted_copy[no_sec_mask]["Churn Prediction"]=="Will Churn").mean()*100 if no_sec_mask.sum()>0 else portfolio_churn
        yes_sec_rate= (predicted_copy[yes_sec_mask]["Churn Prediction"]=="Will Churn").mean()*100 if yes_sec_mask.sum()>0 else 10.0
        sec_pool = predicted_copy[no_sec_mask & (predicted_copy["_risk_clean"].str.lower().isin(["high","medium"]))]
        sec_mrr  = pd.to_numeric(sec_pool[mc], errors="coerce").sum() if mc else 0.0
        problems_data.append({
            "id":"PROB-05","category":"Product Stickiness & Billing Friction","icon":"🛡️",
            "priority":"HIGH RISK" if no_sec_rate>=20 else "MEDIUM RISK","color":T["amber"],
            "title":f"Problem 5: Unprotected Accounts ({no_sec_rate:.1f}% Churn Without Security)",
            "problem":f"{len(sec_pool):,} subscribers without Online Security/Tech Support experience {no_sec_rate:.1f}% churn (vs {yes_sec_rate:.1f}% for protected accounts), representing {_flow(sec_mrr)}.",
            "root_cause":"Single-utility subscribers have lower switching friction; unassisted issues cause provider blame.",
            "solution":f"Complimentary 90-Day Digital Defense Bundle Trial for {len(sec_pool):,} unprotected accounts.",
            "action":f"Provision automated security trial invitations to {len(sec_pool):,} unprotected accounts.",
            "target":f"{len(sec_pool):,} accounts lacking security/tech support",
            "impact":f"Raises switching barriers, protecting {_flow(sec_mrr)}.",
            "sla":"Weekly Automated Trigger Cycle",
        })

    present_cats = ["All Strategic Problems & Solutions"] + sorted({p["category"] for p in problems_data})
    selected_cat = st.radio("Filter by Strategic Domain:", present_cats, horizontal=True, key="retention_filter")

    filtered_probs = [p for p in problems_data if selected_cat.startswith("All") or p["category"]==selected_cat]

    for pb in filtered_probs:
        badge_cls = "status-crimson" if "CRITICAL" in pb["priority"] else "status-amber"
        card = f"""<div style="background:{T['card']};border:1px solid {T['border']};border-left:5px solid {pb['color']};border-radius:10px;padding:1.4rem 1.6rem;margin-bottom:1.1rem;box-shadow:0 2px 6px rgba(15,23,42,0.04);">
<div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:0.6rem;margin-bottom:0.9rem;padding-bottom:0.6rem;border-bottom:1px solid {T['border_sub']};">
<div style="display:flex;align-items:center;gap:0.65rem;">
<span style="font-size:1.4rem;">{pb['icon']}</span>
<span style="color:{T['text']} !important;font-family:'Noto Serif',serif;font-weight:800;font-size:1.12rem;">{escape(pb['title'])}</span>
</div>
<div style="display:flex;align-items:center;gap:0.6rem;">
<span style="font-size:0.75rem;background:{T['surface_sub']};color:{T['text_sec']} !important;padding:3px 8px;border-radius:4px;font-weight:700;border:1px solid {T['border_sub']};">{escape(pb['category'])}</span>
<span class="status-pill {badge_cls}" style="font-weight:800;letter-spacing:0.04em;">{pb['priority']}</span>
</div>
</div>
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:1rem;margin-bottom:0.9rem;">
<div style="background:#fffcfb;border:1px solid #fed7aa;border-radius:8px;padding:1rem 1.15rem;">
<strong style="color:#9a3412 !important;font-size:0.85rem;text-transform:uppercase;letter-spacing:0.05em;">⚠️ Identified Business Problem</strong>
<div style="color:{T['text']} !important;font-size:0.9rem;line-height:1.55;margin:0.4rem 0 0.5rem;font-weight:600;">{escape(pb['problem'])}</div>
<div style="border-top:1px dashed #fdba74;padding-top:0.5rem;">
<strong style="color:#c2410c !important;font-size:0.82rem;text-transform:uppercase;">Root Cause:</strong>
<div style="color:{T['text_sec']} !important;font-size:0.87rem;line-height:1.5;margin-top:0.2rem;">{escape(pb['root_cause'])}</div>
</div>
</div>
<div style="background:#f8fafc;border:1px solid #cbd5e1;border-radius:8px;padding:1rem 1.15rem;">
<strong style="color:#0f766e !important;font-size:0.85rem;text-transform:uppercase;letter-spacing:0.05em;">💡 Possible Strategic Solution</strong>
<div style="color:{T['text']} !important;font-size:0.9rem;line-height:1.55;margin:0.4rem 0 0.5rem;font-weight:600;">{escape(pb['solution'])}</div>
<div style="border-top:1px dashed #cbd5e1;padding-top:0.5rem;">
<strong style="color:#0369a1 !important;font-size:0.82rem;text-transform:uppercase;">Tactical Action:</strong>
<div style="color:{T['text_sec']} !important;font-size:0.87rem;line-height:1.5;margin-top:0.2rem;">{escape(pb['action'])}</div>
</div>
</div>
</div>
<div style="background:{T['surface_sub']};border:1px solid {T['border_sub']};border-radius:6px;padding:0.65rem 1rem;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:0.8rem;font-size:0.84rem;">
<div><strong style="color:{T['text']} !important;">🎯 Target Cohort:</strong> <span style="color:{T['text_sec']} !important;">{escape(pb['target'])}</span></div>
<div><strong style="color:{T['green']} !important;">💰 Capital Insulated:</strong> <span style="color:{T['text_sec']} !important;">{escape(pb['impact'])}</span></div>
<div><strong style="color:{T['text']} !important;">⏱️ SLA:</strong> <span style="color:{T['text_muted']} !important;">{escape(pb['sla'])}</span></div>
</div>
</div>"""
        st.markdown("\n".join(l.strip() for l in card.strip().splitlines() if l.strip()),
                    unsafe_allow_html=True)

    st.markdown("---")
    info_box(
        "⚠️ <strong>Disclaimer:</strong> All retention recommendations are synthesized from customer attributes, "
        "SQLite cohort queries, and predicted churn probabilities. They constitute data-driven strategic guidance, "
        "not guaranteed outcomes. Actual retention depends on execution rigor and subscriber responsiveness."
    )
