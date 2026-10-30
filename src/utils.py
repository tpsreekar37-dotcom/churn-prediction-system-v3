"""
utils.py
--------
Shared utility functions for the Customer Churn Prediction System.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

# ──────────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────────

def get_logger(name: str = "churn") -> logging.Logger:
    """Return a configured logger instance."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(name)


logger = get_logger()

# ──────────────────────────────────────────────────────────────────────────────
# PATH HELPERS
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def path(*parts: str) -> str:
    """Resolve a path relative to project root."""
    return os.path.join(BASE_DIR, *parts)


DATA_RAW        = path("data", "raw")
DATA_PROCESSED  = path("data", "processed")
MODELS_DIR      = path("models")
REPORTS_DIR     = path("reports", "figures")
SQL_DIR         = path("sql")


def ensure_dirs() -> None:
    """Create output directories if they don't exist."""
    for d in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, SQL_DIR]:
        os.makedirs(d, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# PLOT HELPERS
# ──────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "primary":   "#C59B27",   # Champagne Gold
    "secondary": "#B8860B",   # Deep Burnished Gold
    "accent":    "#D4AF37",   # Radiant Gold
    "positive":  "#059669",   # Emerald (Retained / Low Risk)
    "negative":  "#DC2626",   # Crimson (Critical / High Risk)
    "neutral":   "#64748B",   # Muted Stone Slate
}

CHURN_COLORS = [PALETTE["positive"], PALETTE["negative"]]

# ──────────────────────────────────────────────────────────────────────────────
# RISK BANDING
# ──────────────────────────────────────────────────────────────────────────────
# Single source of truth. Every module (predict, generic pipeline, SQL, dashboard)
# must use these thresholds so the same probability never maps to two labels.
RISK_HIGH = 0.65
RISK_MEDIUM = 0.40

# Decision threshold applied to model probabilities. The production model is
# trained on SMOTE-balanced data, so 0.50 over-predicts churn on real traffic.
DECISION_THRESHOLD = 0.50


def classify_risk(probability: float) -> str:
    """Map a churn probability (0-1) to a single canonical risk label."""
    p = safe_float(probability, 0.0)
    if p >= RISK_HIGH:
        return "High"
    if p >= RISK_MEDIUM:
        return "Medium"
    return "Low"


def classify_risk_emoji(probability: float) -> str:
    """Same banding as classify_risk, rendered with the dashboard's status dots."""
    return {"High": "🔴 High", "Medium": "🟡 Medium", "Low": "🟢 Low"}[
        classify_risk(probability)
    ]


# ──────────────────────────────────────────────────────────────────────────────
# MODEL SCHEMA GUARD
# ──────────────────────────────────────────────────────────────────────────────
# The production model was trained on IBM Telco columns. Scoring an upload that
# lacks these columns would silently fill every feature with 0 and return an
# identical probability for every row, so callers must check compatibility first.
TELCO_REQUIRED_COLUMNS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract",
    "InternetService",
    "PaymentMethod",
]


def is_telco_schema(df: pd.DataFrame) -> bool:
    """Return True only if df contains the columns the production model expects."""
    if df is None or not hasattr(df, "columns"):
        return False
    normalised = {str(c).strip().lower().replace("_", "") for c in df.columns}
    return all(
        col.lower().replace("_", "") in normalised for col in TELCO_REQUIRED_COLUMNS
    )

def set_style() -> None:
    """Apply Alabaster Sovereign design system style to matplotlib."""
    plt.rcParams.update({
        "figure.facecolor":  "#FAFAF7",
        "axes.facecolor":    "#FFFFFF",
        "axes.edgecolor":    "#E6D5B8",
        "axes.labelcolor":   "#0F172A",
        "xtick.color":       "#64748B",
        "ytick.color":       "#64748B",
        "text.color":        "#0F172A",
        "grid.color":        "#F0E7D8",
        "grid.alpha":        0.8,
        "font.family":       "sans-serif",
        "axes.titlesize":    14,
        "axes.labelsize":    11,
        "figure.titlesize":  16,
    })


def save_figure(fig: plt.Figure, filename: str) -> str:
    """Save figure to reports/figures/ and return path."""
    out = os.path.join(REPORTS_DIR, filename)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    logger.info(f"Saved figure → {out}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# METRIC & NUMERICAL HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def safe_float(v, default=0.0) -> float:
    """Safely convert any value to float, handling None, NaN, inf, or formatting."""
    if v is None or pd.isna(v):
        return default
    try:
        res = float(v)
        return default if np.isnan(res) or np.isinf(res) else res
    except Exception:
        return default


def safe_int(v, default=0) -> int:
    """Safely convert any value to int, handling None, NaN, inf, or formatting."""
    if v is None or pd.isna(v):
        return default
    try:
        res = float(v)
        return default if np.isnan(res) or np.isinf(res) else int(res)
    except Exception:
        return default


def find_col(df: pd.DataFrame, candidates: list, default=None, exact: bool = False):
    """
    Find the first matching column name in df from a list of candidates.

    By default matching is loose (substring, punctuation-insensitive) which is
    useful for messy uploads. Pass ``exact=True`` for safety-critical lookups
    where a substring hit would be wrong (e.g. resolving a customer ID column).
    """
    if df is None or not hasattr(df, "columns"):
        return default
    for c in df.columns:
        c_clean = str(c).strip().lower().replace("_", "").replace(" ", "")
        for cand in candidates:
            cand_clean = cand.lower().replace("_", "").replace(" ", "")
            if c_clean == cand_clean or (not exact and cand_clean in c_clean):
                return c
    return default


# ──────────────────────────────────────────────────────────────────────────────
# REVENUE / VALUE-AT-RISK HELPERS
# ──────────────────────────────────────────────────────────────────────────────
# Datasets use wildly different names for the money column. Telco has
# ``MonthlyCharges``; a bank dataset has ``balance`` and ``estimated_salary``;
# others use ``amount`` / ``ltv``. If we cannot find one we must say so instead of
# silently reporting $0 of exposure.
REVENUE_COLUMN_CANDIDATES = [
    # (normalised name, is_monthly, human label)
    ("monthlycharges",    True,  "Monthly Charges"),
    ("monthlycharge",     True,  "Monthly Charges"),
    ("mrr",               True,  "Monthly Recurring Revenue"),
    ("charges",           True,  "Charges"),
    ("charge",            True,  "Charges"),
    ("balance",           False, "Account Balance"),
    ("accountbalance",    False, "Account Balance"),
    ("estimatedsalary",   False, "Estimated Salary"),
    ("salary",            False, "Salary"),
    ("annualincome",      False, "Annual Income"),
    ("income",            False, "Income"),
    ("amount",            False, "Amount"),
    ("revenue",           False, "Revenue"),
    ("ltv",               False, "Lifetime Value"),
    ("clv",               False, "Customer Lifetime Value"),
    ("customervalue",     False, "Customer Value"),
]


def find_revenue_column(df: pd.DataFrame):
    """
    Locate the monetary column that best represents value at risk.

    Returns ``(column, is_monthly, label)`` or ``(None, False, None)`` when the
    frame has no usable money column.

    Exact (punctuation-insensitive) matches are preferred so ``TotalCharges``
    never wins over ``MonthlyCharges``. ``balance`` is matched before the looser
    ``amount``/``value`` substring rules for the same reason.
    """
    if df is None or not hasattr(df, "columns") or len(df.columns) == 0:
        return None, False, None

    normalised = {str(c).strip().lower().replace("_", "").replace(" ", ""): c for c in df.columns}

    for cand, is_monthly, label in REVENUE_COLUMN_CANDIDATES:
        if cand in normalised:
            col = normalised[cand]
            # Guard against all-zero / all-null columns (e.g. an empty balance field)
            numeric = pd.to_numeric(df[col], errors="coerce")
            if numeric.notna().any() and numeric.abs().sum() > 0:
                return col, is_monthly, label

    # Loose fallback: a column whose name merely *contains* a money keyword.
    for cand, is_monthly, label in REVENUE_COLUMN_CANDIDATES:
        for norm_name, col in normalised.items():
            if cand in norm_name and "total" not in norm_name:
                numeric = pd.to_numeric(df[col], errors="coerce")
                if numeric.notna().any() and numeric.abs().sum() > 0:
                    return col, is_monthly, label

    return None, False, None


def get_revenue_metrics(predicted: pd.DataFrame, sql_results: dict = None) -> dict:
    """
    Calculate value-at-risk, annualised exposure, ARPU and high-value churners.

    The numbers are derived from the scored DataFrame itself (which still holds
    the original money column) rather than from SQL aggregates. That guarantees
    the figure on screen matches the dataset that was just scored and avoids the
    old failure mode where a missing ``MonthlyCharges`` column was stored as 0
    and every page displayed a confident ``$0``.

    Returned keys:
        at_risk        value held by predicted churners (one period)
        annual         annualised exposure (x12 only when the column is monthly)
        mrr / arr      aliases kept for backwards compatibility
        arpu           mean value across all customers
        churner_arpu   mean value across predicted churners
        high_val       count of high-risk customers
        total          rows scored
        available      False when no monetary column exists → render "N/A"
        label          human name of the money column
        is_monthly     whether the column is a monthly figure
    """
    empty = {
        "at_risk": 0.0, "annual": 0.0, "mrr": 0.0, "arr": 0.0,
        "arpu": 0.0, "churner_arpu": 0.0,
        "high_val": 0, "total": 0,
        "available": False, "label": None, "is_monthly": False,
    }
    if predicted is None or predicted.empty:
        return empty

    predicted = predicted.copy()
    total = len(predicted)

    churn_mask = (
        predicted["Churn Prediction"].astype(str).str.strip().str.lower() == "will churn"
        if "Churn Prediction" in predicted.columns
        else pd.Series(False, index=predicted.index)
    )
    risk_col = "Risk Level" if "Risk Level" in predicted.columns else "risk_level"
    high_mask = (
        predicted[risk_col].astype(str).str.contains("High", case=False, na=False)
        if risk_col in predicted.columns
        else pd.Series(False, index=predicted.index)
    )

    col, is_monthly, label = find_revenue_column(predicted)
    if col is None:
        return {**empty, "total": total, "high_val": int(high_mask.sum())}

    values = pd.to_numeric(predicted[col], errors="coerce").fillna(0.0)
    at_risk = float(values[churn_mask].sum())
    annual = at_risk * 12.0 if is_monthly else at_risk
    arpu = float(values.mean()) if total else 0.0
    churner_arpu = float(values[churn_mask].mean()) if churn_mask.any() else 0.0

    return {
        "at_risk": at_risk, "annual": annual,
        "mrr": at_risk, "arr": annual,
        "arpu": arpu, "churner_arpu": churner_arpu,
        "high_val": int(high_mask.sum()), "total": total,
        "available": True, "label": label, "is_monthly": is_monthly,
    }



def print_separator(title: str = "") -> None:
    """Print a formatted separator line."""
    width = 70
    if title:
        padding = (width - len(title) - 2) // 2
        print("-" * padding + f" {title} " + "-" * padding)
    else:
        print("-" * width)

