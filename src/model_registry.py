"""
model_registry.py
-----------------
Single source of truth for model artifacts, dynamic model selection,
and the shared prediction / evaluation pipeline used across all dashboard pages.

All Streamlit pages must use the functions here rather than loading models
or calling predict_batch independently. This guarantees that the same model
artifact, preprocessing pipeline, decision threshold, and metric calculations
are used everywhere so results are always consistent.
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    logger, MODELS_DIR, DECISION_THRESHOLD,
    classify_risk, classify_risk_emoji, is_telco_schema, find_col,
)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  KNOWN MODEL REGISTRY
#     Maps every friendly name → filename on disk.
#     Ordering here determines the priority of automatic selection.
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY: dict[str, str] = {
    "Voting Ensemble":    "voting_ensemble.pkl",
    "Random Forest":      "random_forest.pkl",
    "XGBoost":            "xgboost.pkl",
    "LightGBM":           "lightgbm.pkl",
    "Logistic Regression":"logistic_regression.pkl",
    "SVM":                "svm.pkl",
}

MODEL_DESCRIPTIONS: dict[str, str] = {
    "Voting Ensemble":    "Soft-voting ensemble combining all individual models for maximum generalization.",
    "Random Forest":      "Ensemble of 200 balanced decision trees. Robust to noise and overfitting.",
    "XGBoost":            "Gradient-boosted trees with 200 estimators, scale_pos_weight=3 for imbalanced classes.",
    "LightGBM":           "Microsoft's gradient boosting with 200 estimators, native imbalance handling.",
    "Logistic Regression":"Linear baseline with L2 regularization (C=1.0). Fast and interpretable.",
    "SVM":                "RBF kernel SVM with probability calibration and balanced class weights.",
}

# In-memory cache so we don't reload the same .pkl on every Streamlit rerun
_model_cache: dict[str, object] = {}


def available_models() -> dict[str, str]:
    """Return only models whose .pkl files actually exist on disk."""
    return {
        name: fname
        for name, fname in MODEL_REGISTRY.items()
        if os.path.exists(os.path.join(MODELS_DIR, fname))
    }


def load_model_by_name(name: str):
    """
    Load and cache a model by its friendly name.
    Raises FileNotFoundError when the artifact is missing.
    """
    if name in _model_cache:
        return _model_cache[name]
    fname = MODEL_REGISTRY.get(name)
    if fname is None:
        raise ValueError(f"Unknown model name: '{name}'. Available: {list(MODEL_REGISTRY)}")
    path = os.path.join(MODELS_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model artifact not found: {path}. "
            "Run 'python main.py' to train and save all model artifacts."
        )
    model = joblib.load(path)
    _model_cache[name] = model
    logger.info(f"Loaded model '{name}' from {path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SAVED VALIDATION METRICS
#     These were computed by train_model.py on the held-out test split.
#     We use them for model selection when the active dataset has no labels.
# ─────────────────────────────────────────────────────────────────────────────

def load_saved_model_results() -> list[dict]:
    """Return saved model results from models/model_results.json, or []."""
    p = os.path.join(MODELS_DIR, "model_results.json")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def get_saved_metrics(model_name: str) -> dict | None:
    """Return saved validation metrics dict for a given model name, or None."""
    for r in load_saved_model_results():
        if r.get("model") == model_name:
            return r
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 2b. CANONICAL DISPLAY METRICS
#     Every page must render per-model scores from this table. It is the only
#     place that decides which numbers are "the" numbers, so Dataset Import,
#     Model Comparison and Model Testing cannot drift apart.
# ─────────────────────────────────────────────────────────────────────────────

def metric_rows(results: list[dict]) -> list[dict]:
    """Map any ``[{model, accuracy, ...}]`` list onto the display schema."""
    rows = []
    for r in results or []:
        name = r.get("model")
        if not name:
            continue
        rows.append({
            "Model":     name,
            "Accuracy":  float(r.get("accuracy") or 0.0),
            "Precision": float(r.get("precision") or 0.0),
            "Recall":    float(r.get("recall") or 0.0),
            "F1-Score":  float(r.get("f1") or 0.0),
            "ROC-AUC":   float(r.get("roc_auc") or 0.0),
        })
    return rows


def canonical_metric_rows(
    pipeline_result: dict | None = None,
    available: dict | None = None,
    use_pipeline: bool = False,
) -> tuple[list[dict], str, str]:
    """
    Return ``(rows, source_note, source_kind)`` — the per-model metrics shown
    anywhere in the application.

    ``use_pipeline=True`` means the active dataset is non-Telco and was scored by
    the generic pipeline. In that case the live holdout metrics from that
    pipeline are authoritative; otherwise the saved Telco validation results are.

    ``available`` limits the rows to models whose artifacts exist on disk.
    """
    if use_pipeline and pipeline_result and pipeline_result.get("results"):
        rows = metric_rows(pipeline_result["results"])
        note = (
            "Metrics computed on the held-out split of the active dataset by the "
            "generic pipeline (models trained on this upload). Identical values are "
            "used on every page."
        )
        return rows, note, "generic"

    saved = load_saved_model_results()
    if available:
        saved = [r for r in saved if r.get("model") in available]
    rows = metric_rows(saved)
    note = (
        "Metrics from the saved held-out test split computed during training "
        "(models/model_results.json). Identical values are used on every page."
    )
    return rows, note, "saved"


# ─────────────────────────────────────────────────────────────────────────────
# 3.  DYNAMIC MODEL SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def select_best_model(df: pd.DataFrame, target_col: str | None = None) -> dict:
    """
    Dynamically select the best model for the given dataset.

    Strategy:
      • If labels are available (target_col is not None):
          Use the saved validation ROC-AUC (from training time, computed on the
          held-out test split — NOT on the dataset supplied here) to rank
          eligible models.  We deliberately do not evaluate on the supplied
          dataset to avoid selection bias (choosing the model that happens to fit
          this particular slice of data best).
      • If no labels:
          Same — rank by saved ROC-AUC.
      • Compatibility check: a model is only "eligible" if its required features
          can be satisfied by the supplied dataframe (Telco schema check for the
          production models).

    Returns a dict:
        name          – chosen model name
        model         – loaded model object
        metric_name   – selection metric label
        metric_value  – selection metric value
        reason        – human-readable explanation
        all_scores    – list of (name, auc) for all eligible models
    """
    avail = available_models()
    if not avail:
        raise RuntimeError("No trained model artifacts found. Run 'python main.py' first.")

    is_telco = is_telco_schema(df)
    saved = load_saved_model_results()

    # Build a ranking table
    scores = []
    for name in avail:
        # Compatibility: Telco models work only on Telco schema.
        # Voting Ensemble depends on the same preprocessing so same restriction.
        compatible = is_telco  # all production artifacts share the Telco preprocessing
        if not compatible:
            continue  # skip; caller should use generic_pipeline for non-Telco data
        row = next((r for r in saved if r.get("model") == name), None)
        if row:
            scores.append((name, float(row.get("roc_auc", 0)), float(row.get("f1", 0))))
        else:
            scores.append((name, 0.0, 0.0))

    if not scores:
        # Non-Telco schema — fall back to the first available artifact
        fallback_name = next(iter(avail))
        model = load_model_by_name(fallback_name)
        return {
            "name":         fallback_name,
            "model":        model,
            "metric_name":  "saved ROC-AUC",
            "metric_value": None,
            "reason":       "Dataset schema does not match Telco training schema. "
                            "Falling back to best available artifact; use the generic pipeline "
                            "or upload a Telco-compatible dataset for proper model selection.",
            "all_scores":   [],
        }

    # Sort by ROC-AUC then F1
    scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
    best_name, best_auc, best_f1 = scores[0]
    model = load_model_by_name(best_name)

    reason = (
        f"Selected '{best_name}' based on the highest saved validation ROC-AUC "
        f"({best_auc:.4f}) computed on the held-out test split during training "
        f"(F1={best_f1:.4f}). Selection uses pre-existing validation evidence, "
        "not the currently supplied dataset, to avoid selection bias."
    )

    return {
        "name":         best_name,
        "model":        model,
        "metric_name":  "Saved Validation ROC-AUC",
        "metric_value": best_auc,
        "reason":       reason,
        "all_scores":   [(n, auc, f1) for n, auc, f1 in scores],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SHARED PREDICTION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def run_predictions(
    df: pd.DataFrame,
    model_name: str | None = None,
    model=None,
    threshold: float = DECISION_THRESHOLD,
) -> pd.DataFrame:
    """
    Central prediction function used by ALL pages.

    Args:
        df:           Raw input DataFrame.
        model_name:   Friendly model name (e.g. "Random Forest").
                      Ignored when ``model`` is supplied directly.
        model:        Pre-loaded model object (optional, avoids redundant disk IO).
        threshold:    Decision threshold (defaults to DECISION_THRESHOLD from utils).

    Returns:
        df with columns added:
          - Churn Probability  (0.0 – 100.0)
          - Churn Prediction   ("Will Churn" / "Will Stay")
          - Risk Level         ("🔴 High" / "🟡 Medium" / "🟢 Low")
          - churn_probability  (0.0 – 1.0, raw)
          - churn_prediction   (0 / 1, raw)
          - risk_level         ("High" / "Medium" / "Low")

    Raises ValueError when the dataset is incompatible with the Telco model
    AND no target column is available to run the generic pipeline.
    """
    from src.predict import predict_batch, preprocess_dataframe
    from src.generic_pipeline import run_generic_pipeline, predict_generic_batch

    if model is None:
        if model_name is None:
            # Auto-select
            sel = select_best_model(df)
            model = sel["model"]
            model_name = sel["name"]
        else:
            model = load_model_by_name(model_name)

    # Route: Telco schema → production model; anything else → generic pipeline
    if is_telco_schema(df):
        return predict_batch(df, model=model, require_schema=True)
    else:
        # Try to find a target column for the generic pipeline
        target_candidates = [
            c for c in df.columns
            if any(k in c.lower() for k in ["churn", "target", "outcome", "attrition", "exited"])
        ]
        if target_candidates:
            target_col = target_candidates[0]
            id_col = find_col(df, ["customerid", "customer_id", "custid", "accountid", "id"],
                              default=None, exact=True)
            pipeline = run_generic_pipeline(df, target_col, id_col)
            return predict_generic_batch(df, pipeline, target_col, id_col)
        else:
            raise ValueError(
                "Dataset does not match the Telco schema and has no detectable churn/target column. "
                "Add a binary churn column or upload a Telco-compatible dataset."
            )


# ─────────────────────────────────────────────────────────────────────────────
# 5.  SHARED EVALUATION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_predictions(
    pred_df: pd.DataFrame,
    target_col: str,
    threshold: float = DECISION_THRESHOLD,
) -> dict:
    """
    Compute all classification metrics for a predictions DataFrame that
    already contains ground-truth labels.

    The convention used here is the single authoritative definition for
    the whole application:
        TP = actual churn      AND predicted churn
        TN = actual retention  AND predicted retention
        FP = actual retention  BUT predicted churn
        FN = actual churn      BUT predicted retention

    Args:
        pred_df:    DataFrame returned by run_predictions (or similar) that
                    still contains the original target_col.
        target_col: Name of the column with ground-truth labels.
        threshold:  Decision threshold that was applied to produce Churn Prediction.

    Returns dict with:
        accuracy, precision, recall, f1, roc_auc (or None),
        tp, tn, fp, fn,
        actual_bin   (numpy array 0/1),
        pred_bin     (numpy array 0/1),
        probs        (numpy array 0..1),
        n_total, n_churn_actual, n_churn_predicted,
        churn_rate_actual, churn_rate_predicted,
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score,
    )

    # Normalise actual labels to 0/1
    actual_raw = pred_df[target_col].astype(str).str.strip().str.lower()
    actual_bin = actual_raw.apply(
        lambda x: 1 if x in {"1", "1.0", "yes", "true", "churn", "will churn", "churned"} else 0
    ).values

    # Normalise predicted labels to 0/1
    pred_raw = pred_df["Churn Prediction"].astype(str).str.strip().str.lower()
    pred_bin = pred_raw.apply(
        lambda x: 1 if x in {"will churn", "churn", "1", "1.0", "yes", "true"} else 0
    ).values

    # Probabilities
    probs = pred_df["churn_probability"].values if "churn_probability" in pred_df.columns else (
        pred_df["Churn Probability"].values / 100.0
        if "Churn Probability" in pred_df.columns else np.zeros(len(pred_df))
    )
    if probs.max() > 1.0:
        probs = probs / 100.0

    n = len(actual_bin)
    tp = int(((actual_bin == 1) & (pred_bin == 1)).sum())
    tn = int(((actual_bin == 0) & (pred_bin == 0)).sum())
    fp = int(((actual_bin == 0) & (pred_bin == 1)).sum())
    fn = int(((actual_bin == 1) & (pred_bin == 0)).sum())

    acc  = round(accuracy_score(actual_bin, pred_bin), 4)
    prec = round(precision_score(actual_bin, pred_bin, zero_division=0), 4)
    rec  = round(recall_score(actual_bin, pred_bin, zero_division=0), 4)
    f1   = round(f1_score(actual_bin, pred_bin, zero_division=0), 4)

    try:
        auc = round(roc_auc_score(actual_bin, probs), 4)
    except Exception:
        auc = None

    return {
        "accuracy":           acc,
        "precision":          prec,
        "recall":             rec,
        "f1":                 f1,
        "roc_auc":            auc,
        "tp":                 tp,
        "tn":                 tn,
        "fp":                 fp,
        "fn":                 fn,
        "actual_bin":         actual_bin,
        "pred_bin":           pred_bin,
        "probs":              probs,
        "n_total":            n,
        "n_churn_actual":     int(actual_bin.sum()),
        "n_churn_predicted":  int(pred_bin.sum()),
        "churn_rate_actual":  round(float(actual_bin.mean()) * 100, 2) if n > 0 else 0.0,
        "churn_rate_predicted": round(float(pred_bin.mean()) * 100, 2) if n > 0 else 0.0,
        "threshold":          threshold,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6.  DATASET VALIDATION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Quick validation pass over an uploaded DataFrame.

    Excel files often contain phantom empty columns/rows from cell formatting.
    We strip those before reporting counts so users don't see false missing-value
    warnings for data that is actually clean.

    Returns:
        n_rows, n_cols, n_missing, n_duplicates,
        has_target, target_col,
        is_telco, schema_note,
        warnings  (list of str)
    """
    # ── Drop columns that are entirely NaN (Excel ghost columns) ──────────────
    df_clean = df.dropna(axis=1, how="all")
    # ── Drop rows that are entirely NaN (Excel ghost rows at the bottom) ──────
    df_clean = df_clean.dropna(axis=0, how="all")

    n_rows = len(df_clean)
    n_cols = len(df_clean.columns)
    # Count meaningful missing values: cells that are NaN in real data columns
    n_missing = int(df_clean.isnull().sum().sum())
    n_duplicates = int(df_clean.duplicated().sum())

    target_candidates = [
        c for c in df_clean.columns
        if any(k in c.lower() for k in ["churn", "target", "outcome", "attrition", "exited"])
    ]
    has_target = bool(target_candidates)
    target_col = target_candidates[0] if has_target else None

    is_telco = is_telco_schema(df_clean)
    schema_note = (
        "Telco-compatible schema detected. Production models can be applied directly."
        if is_telco
        else "Non-Telco schema. Generic pipeline will be used if a target column is present."
    )

    warns = []
    if n_missing > 0:
        warns.append(f"{n_missing:,} missing values detected in data columns — will be imputed during preprocessing.")
    if n_duplicates > 0:
        warns.append(f"{n_duplicates:,} duplicate rows detected — consider deduplication.")
    if n_rows < 50:
        warns.append(f"Small dataset ({n_rows} rows) — metrics may be unreliable.")
    if not has_target:
        warns.append("No churn/target column detected — supervised metrics unavailable.")

    return {
        "n_rows":       n_rows,
        "n_cols":       n_cols,
        "n_missing":    n_missing,
        "n_duplicates": n_duplicates,
        "has_target":   has_target,
        "target_col":   target_col,
        "is_telco":     is_telco,
        "schema_note":  schema_note,
        "warnings":     warns,
    }
