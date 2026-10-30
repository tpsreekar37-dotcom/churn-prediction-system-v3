"""
generic_pipeline.py
--------------------
A fully schema-agnostic ML pipeline for universal churn prediction.
Works on ANY tabular customer dataset — not just telecom.

Steps:
  1. Auto-detect column types (numeric, categorical, date)
  2. Impute missing values
  3. Encode categoricals (binary → 0/1, multi → label encode)
  4. Scale numeric features
  5. Apply SMOTE if class imbalance exists
  6. Train Random Forest and XGBoost
  7. Save the model with the highest holdout ROC-AUC as the production model
  8. Return predictions + feature importances + metrics

Usage:
  from src.generic_pipeline import run_generic_pipeline, predict_generic
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import classify_risk_emoji, DECISION_THRESHOLD


# ──────────────────────────────────────────────────────────────────────────────
# 1. COLUMN TYPE DETECTION
# ──────────────────────────────────────────────────────────────────────────────

def detect_column_types(df: pd.DataFrame, target_col: str, id_col: str = None):
    """
    Auto-detect which columns are numeric, categorical, and date.
    Returns a dict: {col_name: "numeric" | "categorical" | "date" | "drop"}
    """
    col_types = {}
    for col in df.columns:
        if col == target_col:
            col_types[col] = "target"
            continue
        if id_col and col == id_col:
            col_types[col] = "drop"
            continue

        # Prevent target data leakage: drop other churn/outcome columns from X
        col_lower = str(col).strip().lower()
        if ("churn" in col_lower or "attrition" in col_lower or 
            col_lower in ["exited", "customer status", "target", "outcome"]):
            col_types[col] = "drop"
            continue

        dtype = df[col].dtype
        n_unique = df[col].nunique(dropna=True)

        # Date detection
        if dtype == object:
            sample = df[col].dropna().head(50).astype(str)
            date_like = sample.str.match(
                r"^\d{1,4}[-/]\d{1,2}[-/]\d{2,4}$|^\d{4}-\d{2}-\d{2}$"
            ).sum()
            if date_like > len(sample) * 0.7:
                col_types[col] = "date"
                continue

            # Check if object column is actually numeric (e.g. TotalCharges with spaces or formatted numbers)
            clean_str = df[col].astype(str).str.strip().str.replace("$", "", regex=False).str.replace(",", "", regex=False)
            clean_str = clean_str.replace(["", "nan", "none", "null"], np.nan)
            num_converted = pd.to_numeric(clean_str, errors="coerce")
            if num_converted.notna().sum() > len(df[col].dropna()) * 0.75:
                if num_converted.nunique() <= 2:
                    col_types[col] = "binary"
                else:
                    col_types[col] = "numeric"
                continue

        if pd.api.types.is_numeric_dtype(dtype):
            if n_unique <= 2:
                col_types[col] = "binary"
            else:
                col_types[col] = "numeric"
        elif dtype == object or isinstance(dtype, pd.CategoricalDtype):
            if n_unique <= 2:
                col_types[col] = "binary"
            elif n_unique > 50:
                col_types[col] = "drop"   # too high cardinality → drop
            else:
                col_types[col] = "categorical"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            col_types[col] = "date"
        else:
            col_types[col] = "numeric"

    return col_types


# ──────────────────────────────────────────────────────────────────────────────
# 2. PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────

def normalize_target(series: pd.Series) -> pd.Series:
    """
    Convert any binary target column to 0/1 integers.
    Handles: Yes/No, True/False, 1/0, Churned/Active, floats with NaNs/infs, string variations.
    The churn/positive class is reliably mapped to 1.
    """
    # Try numeric conversion first
    num_series = pd.to_numeric(series, errors="coerce")
    if num_series.dropna().nunique() == 2:
        max_val = num_series.dropna().max()
        return (num_series == max_val).astype(int)

    # String normalization: strip whitespace and lowercase
    str_series = series.astype(str).str.strip().str.lower()
    valid_mask = ~str_series.isin(["nan", "none", "null", ""])
    unique_vals = str_series[valid_mask].unique()

    positive_hints = {"yes", "true", "churn", "churned", "1", "1.0", "left", "cancelled",
                      "inactive", "attrition", "exited", "lost", "y"}

    if len(unique_vals) == 2:
        pos_val = None
        for v in unique_vals:
            if v in positive_hints:
                pos_val = v
                break
        if pos_val is None:
            pos_val = sorted(unique_vals)[-1]
        return (str_series == pos_val).astype(int)

    # If any positive hint appears in the series
    has_pos = str_series.isin(positive_hints)
    if has_pos.sum() > 0:
        return has_pos.astype(int)

    if len(unique_vals) <= 1:
        return pd.Series(0, index=series.index)

    # Multi-value fallback
    return (str_series == unique_vals[1]).astype(int)


def preprocess_generic(df: pd.DataFrame, target_col: str,
                        id_col: str = None) -> tuple:
    """
    Full preprocessing pipeline for any dataset.

    The train/test split happens **before** any fitted transformation (imputation
    values, label encoders, date reference dates, scaler). Every constant is
    learned on the training partition and applied unchanged to the test partition,
    so holdout metrics are not inflated by leakage.

    Returns:
        X_train, X_test, y_train, y_test, scaler, encoders, col_types,
        feature_names, scaler_columns, date_refs, impute_values
    """
    df = df.copy()

    # ── Drop rows where target_col is NaN or infinite
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[target_col])

    # ── Detect column types (dtype-based only — does not look at labels)
    col_types = detect_column_types(df, target_col, id_col)

    # ── Drop high-cardinality / ID / leakage columns
    drop_cols = [c for c, t in col_types.items() if t == "drop"]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    # ── Normalize target
    df[target_col] = normalize_target(df[target_col])
    y_all = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int)

    # ── SPLIT FIRST: everything below is fitted on train_df only
    try:
        train_idx, test_idx = train_test_split(
            df.index, test_size=0.20, random_state=42, stratify=y_all
        )
    except ValueError:
        # A class can be too rare to stratify; fall back to a plain split.
        train_idx, test_idx = train_test_split(
            df.index, test_size=0.20, random_state=42
        )

    train_df = df.loc[train_idx].copy()
    test_df  = df.loc[test_idx].copy()

    # ── Parse date columns → days-since-reference feature.
    # The reference date is the training minimum, so the same calendar day maps
    # to the same integer in train, test and at inference.
    date_cols = [c for c, t in col_types.items() if t == "date" and c in df.columns]
    date_refs = {}
    for col in date_cols:
        try:
            ref = pd.to_datetime(train_df[col], errors="coerce").min()
            if pd.isna(ref):
                raise ValueError("no parseable dates in training partition")
            date_refs[col] = ref
            for frame in (train_df, test_df):
                parsed = pd.to_datetime(frame[col], errors="coerce")
                frame[col + "_days"] = (parsed - ref).dt.days.fillna(0).astype(int)
                frame.drop(columns=[col], inplace=True)
            col_types[col + "_days"] = "numeric"
        except Exception:
            train_df.drop(columns=[col], inplace=True, errors="ignore")
            test_df.drop(columns=[col], inplace=True, errors="ignore")

    feature_cols = [c for c in train_df.columns if c != target_col]

    # ── Imputation parameters (learned on train only)
    impute_values = {}
    for col in feature_cols:
        ctype = col_types.get(col, "numeric")
        if ctype == "numeric":
            val = pd.to_numeric(train_df[col], errors="coerce").median()
            impute_values[col] = 0.0 if pd.isna(val) else float(val)
        else:
            mode = train_df[col].mode()
            impute_values[col] = mode.iloc[0] if not mode.empty else 0

    def _impute(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        for col in feature_cols:
            if col not in frame.columns:
                continue
            ctype = col_types.get(col, "numeric")
            if ctype == "numeric":
                frame[col] = pd.to_numeric(frame[col], errors="coerce").replace(
                    [np.inf, -np.inf], np.nan
                ).fillna(impute_values[col])
            else:
                frame[col] = frame[col].fillna(impute_values[col])
        return frame

    train_df = _impute(train_df)
    test_df  = _impute(test_df)

    # ── Encoders (fit on train only)
    encoders = {}

    binary_cols = [c for c, t in col_types.items()
                   if t == "binary" and c in feature_cols]
    for col in binary_cols:
        unique_vals = train_df[col].dropna().unique()
        if set(map(str, unique_vals)) <= {"0", "1", "0.0", "1.0"}:
            encoders[col] = None  # already numeric
        else:
            le = LabelEncoder()
            le.fit(train_df[col].astype(str))
            encoders[col] = le

    cat_cols = [c for c, t in col_types.items()
                if t == "categorical" and c in feature_cols]
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(train_df[col].astype(str))
        encoders[col] = le

    # Any remaining object column is encoded too.
    remaining_obj = [c for c in feature_cols
                     if pd.api.types.is_object_dtype(train_df[c]) and c not in encoders]
    for col in remaining_obj:
        le = LabelEncoder()
        le.fit(train_df[col].astype(str))
        encoders[col] = le

    def _encode(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        for col, encoder in encoders.items():
            if col not in frame.columns:
                continue
            if encoder is None:
                frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0).astype(int)
            else:
                known = set(encoder.classes_)
                frame[col] = frame[col].astype(str).map(
                    lambda x: int(encoder.transform([x])[0]) if x in known else 0
                )
        return frame

    train_df = _encode(train_df)
    test_df  = _encode(test_df)

    # ── Split X / y
    X_train = train_df.drop(columns=[target_col])
    X_test  = test_df.drop(columns=[target_col])
    y_train = pd.to_numeric(train_df[target_col], errors="coerce").fillna(0).astype(int)
    y_test  = pd.to_numeric(test_df[target_col],  errors="coerce").fillna(0).astype(int)
    feature_names = list(X_train.columns)

    # ── Scale numeric columns (scaler fitted on train only)
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols]  = scaler.transform(X_test[numeric_cols])

    return (X_train, X_test, y_train, y_test, scaler, encoders, col_types,
            feature_names, numeric_cols, date_refs, impute_values)


# ──────────────────────────────────────────────────────────────────────────────
# 3. SMOTE
# ──────────────────────────────────────────────────────────────────────────────

def apply_smote_generic(X_train: pd.DataFrame, y_train: pd.Series):
    """Apply SMOTE only if minority class has enough samples."""
    minority_count = y_train.value_counts().min()
    if minority_count < 6:
        # Not enough samples for SMOTE
        return X_train, y_train

    churn_rate = y_train.mean()
    if churn_rate < 0.35 or churn_rate > 0.65:
        # Class imbalance — apply SMOTE
        k = min(5, minority_count - 1)
        smote = SMOTE(random_state=42, k_neighbors=k)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        X_res = pd.DataFrame(X_res, columns=X_train.columns)
        y_res = pd.Series(y_res, name=y_train.name)
        return X_res, y_res

    return X_train, y_train


# ──────────────────────────────────────────────────────────────────────────────
# 4. MODEL TRAINING
# ──────────────────────────────────────────────────────────────────────────────

def get_generic_models() -> dict:
    """Return all candidate models for the generic pipeline."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=10, random_state=42,
            n_jobs=-1, class_weight="balanced"
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.1,
            eval_metric="logloss", random_state=42, n_jobs=-1,
            scale_pos_weight=3,
        ),
        "SVM": SVC(
            kernel="rbf", probability=True, random_state=42,
            class_weight="balanced",
        ),
    }
    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(
            n_estimators=150, max_depth=8, learning_rate=0.1,
            random_state=42, n_jobs=-1, verbose=-1,
            is_unbalance=True,
        )
    except ImportError:
        pass
    return models


def train_generic_models(X_train, X_test, y_train, y_test) -> tuple:
    """
    Train both candidate models and keep the one with the best holdout ROC-AUC.
    """
    models  = get_generic_models()
    results = []
    trained = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model

        y_proba = model.predict_proba(X_test)[:, 1]
        # Use the same decision threshold the app serves with.
        y_pred  = (y_proba >= DECISION_THRESHOLD).astype(int)

        metrics = {
            "model":     name,
            "accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
            "roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
        }
        results.append(metrics)

    best_row  = max(results, key=lambda r: (r["roc_auc"], r["f1"]))
    best_name = best_row["model"]
    best_model = trained[best_name]

    return best_model, trained, results, best_name


# ──────────────────────────────────────────────────────────────────────────────
# 5. FEATURE IMPORTANCE
# ──────────────────────────────────────────────────────────────────────────────

def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """Extract feature importances from any supported model."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return pd.DataFrame(columns=["Feature", "Importance"])

    return (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )


# ──────────────────────────────────────────────────────────────────────────────
# 6. FULL PIPELINE (called from dashboard)
# ──────────────────────────────────────────────────────────────────────────────

def run_generic_pipeline(df: pd.DataFrame, target_col: str,
                         id_col: str = None) -> dict:
    """
    Full end-to-end pipeline for any uploaded dataset.

    Args:
        df:         Raw uploaded DataFrame
        target_col: Name of the churn/target column (binary)
        id_col:     Name of customer ID column to exclude (optional)

    Returns dict with keys:
        best_model, best_name, results, feature_importance_df,
        X_test, y_test, scaler, encoders, col_types, feature_names,
        churn_rate, n_rows, n_features
    """
    # 1. Preprocess
    (X_train, X_test, y_train, y_test, scaler, encoders, col_types,
     feature_names, scaler_columns, date_refs, impute_values) = \
        preprocess_generic(df, target_col, id_col)

    n_rows     = len(df)
    n_features = len(feature_names)
    churn_rate = y_train.mean()

    # 2. SMOTE
    X_train_bal, y_train_bal = apply_smote_generic(X_train, y_train)

    # 3. Train models
    best_model, trained, results, best_name = train_generic_models(
        X_train_bal, X_test, y_train_bal, y_test
    )

    # 4. Feature importance
    fi_df = get_feature_importance(best_model, feature_names)

    return {
        "best_model":            best_model,
        "all_models":            trained,
        "best_name":             best_name,
        "results":               results,
        "feature_importance_df": fi_df,
        "X_test":                X_test,
        "y_test":                y_test,
        "scaler":                scaler,
        "encoders":              encoders,
        "col_types":             col_types,
        "feature_names":         feature_names,
        "scaler_columns":        scaler_columns,
        "date_refs":             date_refs,
        "impute_values":         impute_values,
        "churn_rate":            churn_rate,
        "n_rows":                n_rows,
        "n_features":            n_features,
        "target_column":          target_col,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 7. BATCH PREDICTION (called from dashboard after training)
# ──────────────────────────────────────────────────────────────────────────────

def predict_generic_batch(df_raw: pd.DataFrame, pipeline_result: dict,
                           target_col: str, id_col: str = None,
                           model=None) -> pd.DataFrame:
    """
    Run predictions on the full uploaded dataset.

    Args:
        df_raw:          Original uploaded frame.
        pipeline_result: Dict returned by :func:`run_generic_pipeline`.
        target_col:      Ground-truth column (kept in the output for evaluation).
        id_col:          Customer identifier (unused except for documentation).
        model:           Optional estimator override, e.g. one of
                         ``pipeline_result["all_models"]`` to score with a
                         non-default model.

    Returns df_raw with added columns: churn_probability, churn_prediction,
    risk_level plus the dashboard-facing ``Churn Probability`` /
    ``Churn Prediction`` / ``Risk Level``.
    """
    best_model     = model if model is not None else pipeline_result["best_model"]
    scaler         = pipeline_result["scaler"]
    encoders       = pipeline_result["encoders"]
    feature_names  = pipeline_result["feature_names"]
    scaler_columns = pipeline_result["scaler_columns"]
    date_refs      = pipeline_result.get("date_refs", {})
    impute_values  = pipeline_result.get("impute_values", {})

    df = df_raw.copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    # Handle any columns added during date processing. Reuse the reference date
    # captured at training time so the same row always produces the same value.
    for col in list(df.columns):
        if col + "_days" in feature_names and col not in feature_names:
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                ref = date_refs.get(col)
                if ref is None:
                    continue
                df[col + "_days"] = (parsed - ref).dt.days.fillna(0).astype(int)
            except Exception:
                pass

    # Keep only feature columns
    X = pd.DataFrame(index=df.index)
    for col in feature_names:
        if col in df.columns:
            X[col] = df[col]
        else:
            X[col] = 0

    X = X.replace([np.inf, -np.inf], np.nan)

    # Impute with the *training* values, never the batch's own statistics.
    for col in X.columns:
        fallback = impute_values.get(col, 0)
        if pd.api.types.is_numeric_dtype(X[col]) or col in scaler_columns:
            X[col] = pd.to_numeric(X[col], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            ).fillna(fallback if not isinstance(fallback, str) else 0)
        else:
            X[col] = X[col].fillna(fallback)

    # Encode with the encoders fitted on the training partition.
    for col, le in encoders.items():
        if col not in X.columns:
            continue
        if le is None:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0).astype(int)
        else:
            known_classes = set(le.classes_)
            X[col] = X[col].apply(
                lambda v: le.transform([str(v)])[0] if str(v) in known_classes else 0
            )

    # Use exact column sequence fitted by StandardScaler
    X[scaler_columns] = scaler.transform(X[scaler_columns])

    # Align feature order
    X = X[feature_names]

    # Predict
    probas = best_model.predict_proba(X)[:, 1]
    preds  = (probas >= DECISION_THRESHOLD).astype(int)

    result_df = df_raw.copy()
    result_df["Churn Probability"]  = (probas * 100).round(1)
    result_df["Churn Prediction"]   = np.where(preds == 1, "Will Churn", "Will Stay")
    result_df["Risk Level"]         = [classify_risk_emoji(p) for p in probas]
    result_df["churn_probability"]  = probas.round(4)
    result_df["churn_prediction"]   = preds
    result_df["risk_level"]         = [classify_risk_emoji(p).split(" ", 1)[1] for p in probas]

    return result_df
