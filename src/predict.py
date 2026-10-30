"""
predict.py
----------
Prediction module for the Customer Churn Prediction System.
Loads the saved best model and provides single-customer and batch prediction.
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    logger, MODELS_DIR,
    RISK_HIGH, RISK_MEDIUM, DECISION_THRESHOLD,
    classify_risk, classify_risk_emoji, is_telco_schema,
)


# ──────────────────────────────────────────────────────────────────────────────
# 1. MODEL LOADING
# ──────────────────────────────────────────────────────────────────────────────

_model_cache = {}
_artifact_cache = None

def load_model(model_name: str = "best_model"):
    """Load model from disk with caching."""
    if model_name not in _model_cache:
        path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found: {path}\n"
                "Run main.py or src/train_model.py first."
            )
        _model_cache[model_name] = joblib.load(path)
        logger.info(f"Loaded model: {path}")
        _warn_if_version_mismatch()
    return _model_cache[model_name]


def _warn_if_version_mismatch() -> None:
    """
    Compare the installed scikit-learn version against the one recorded when the
    artifact was trained. Unpickling across versions can silently change results.
    """
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    if not os.path.exists(meta_path):
        return
    try:
        import sklearn
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        trained_with = str(meta.get("sklearn_version", ""))
        if trained_with and trained_with != sklearn.__version__:
            logger.warning(
                "Model was trained with scikit-learn %s but %s is installed. "
                "Retrain with 'python main.py' to guarantee identical results.",
                trained_with, sklearn.__version__,
            )
    except Exception:
        pass


def get_best_model_name() -> str:
    """Read best model name from text file."""
    p = os.path.join(MODELS_DIR, "best_model_name.txt")
    if os.path.exists(p):
        with open(p) as f:
            return f.read().strip()
    return "best_model"


def load_preprocessing_artifacts():
    """Load the transformations and feature constants saved during training."""
    global _artifact_cache
    if _artifact_cache is None:
        scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
        encoders_path = os.path.join(MODELS_DIR, "encoders.pkl")
        config_path = os.path.join(MODELS_DIR, "feature_config.json")
        missing = [p for p in [scaler_path, encoders_path, config_path] if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                "Preprocessing artifacts are missing. Run 'python main.py' once to train "
                "and create scaler.pkl, encoders.pkl, and feature_config.json."
            )
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        _artifact_cache = (joblib.load(scaler_path), joblib.load(encoders_path), config)
    return _artifact_cache


# ──────────────────────────────────────────────────────────────────────────────
# 2. PREPROCESSING FOR INFERENCE
# ──────────────────────────────────────────────────────────────────────────────

def _clip_to_training_bounds(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Winsorise numeric columns using the limits learned during training.

    Training caps MonthlyCharges/TotalCharges at the 1st/99th percentile; without
    this the model sees out-of-distribution values at inference.
    """
    for col, pair in (config.get("outlier_bounds") or {}).items():
        if col in df.columns and len(pair) == 2:
            df[col] = np.clip(df[col], float(pair[0]), float(pair[1]))
    return df


# Mapping tables mirroring preprocessing.py
BINARY_MAP = {"Yes": 1, "No": 0}

SERVICE_COLS = [
    "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]

CONTRACT_MAP = {"Month-to-month": 0, "One year": 1, "Two year": 2}

GENDER_MAP   = {"Male": 1, "Female": 0}
PAYMENT_MAP  = {
    "Bank transfer (automatic)": 0,
    "Credit card (automatic)": 1,
    "Electronic check": 2,
    "Mailed check": 3,
}

INTERNET_MAP = {"DSL": 0, "Fiber optic": 1, "No": 2}


def preprocess_single(customer: dict) -> pd.DataFrame:
    """
    Accept a dict of raw customer data (as collected from dashboard)
    and return a preprocessed DataFrame ready for model inference.
    """
    row = customer.copy()
    scaler, encoders, config = load_preprocessing_artifacts()

    # ── Binary
    for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        if col in row:
            row[col] = BINARY_MAP.get(str(row[col]), 0)

    # ── Service columns
    for col in SERVICE_COLS:
        if col in row:
            val = str(row[col]).lower()
            row[col] = 0 if val in ["no", "no internet service", "no phone service"] else 1

    # ── Contract
    if "Contract" in row:
        row["Contract"] = CONTRACT_MAP.get(str(row["Contract"]), 0)

    # ── Label-encoded categories.  Use the fitted encoders rather than
    # hand-written maps so category codes cannot drift from training.
    for col in ["gender", "InternetService", "PaymentMethod"]:
        if col in row:
            encoder = encoders.get(col)
            value = str(row[col])
            if encoder is not None and hasattr(encoder, "classes_"):
                row[col] = int(encoder.transform([value])[0]) if value in encoder.classes_ else 0
            else:
                fallback = {"gender": GENDER_MAP, "InternetService": INTERNET_MAP,
                            "PaymentMethod": PAYMENT_MAP}[col]
                row[col] = fallback.get(value, 0)

    # ── Numeric coerce
    for col in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]:
        if col in row:
            row[col] = float(row.get(col, 0))

    df = pd.DataFrame([row])

    # ── Match the training-time winsorisation
    df = _clip_to_training_bounds(df, config)

    # ── Feature engineering (mirroring feature_engineering.py)
    df["TotalCharges"]    = df.get("TotalCharges", df["MonthlyCharges"] * df["tenure"])
    df["ChargesPerTenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)
    # Keep this list identical to feature_engineering.create_features().
    # MultipleLines is deliberately not included in the trained ServiceCount.
    service_count_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies",
    ]
    df["ServiceCount"] = df[[c for c in service_count_cols if c in df.columns]].sum(axis=1)
    df["IsHighValue"]     = (df["MonthlyCharges"] >= config["monthly_charge_threshold"]).astype(int)
    df["HasLoyalty"]      = (df["tenure"] > 24).astype(int)
    contract_risk = (2 - df["Contract"]) / 2
    charges_norm = (df["MonthlyCharges"] - config["monthly_charge_min"]) / (
        config["monthly_charge_max"] - config["monthly_charge_min"] + 1e-9
    )
    tenure_norm = 1 - (df["tenure"] - config["tenure_min"]) / (
        config["tenure_max"] - config["tenure_min"] + 1e-9
    )
    df["RiskScore"] = (0.4 * contract_risk + 0.3 * charges_norm + 0.3 * tenure_norm).round(4)

    # The model was fitted with these three columns standardised.
    scale_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    df[scale_cols] = scaler.transform(df[scale_cols])

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 3. PREDICTION FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def predict_single(customer: dict, model=None) -> dict:
    """
    Predict churn for a single customer.

    Args:
        customer: dict of raw customer features
        model:    optional pre-loaded model (for speed in dashboard)

    Returns:
        dict with keys:
          - churn_prediction (0/1)
          - churn_probability (float 0-1)
          - churn_label       ("Churn" / "No Churn")
          - risk_level        ("High" / "Medium" / "Low")
    """
    if model is None:
        model = load_model("best_model")

    df = preprocess_single(customer)

    # Strip non-feature columns
    _drop = [c for c in df.columns if c.lower() in
             {"churn", "customerid", "customer_id", "custid", "accountid"}]
    if _drop:
        df = df.drop(columns=_drop, errors="ignore")

    # Align with model's expected feature order
    expected_cols = None
    if hasattr(model, "feature_names_in_"):
        expected_cols = list(model.feature_names_in_)
    elif hasattr(model, "estimators_") and model.estimators_:
        for _est in model.estimators_:
            if hasattr(_est, "feature_names_in_"):
                expected_cols = list(_est.feature_names_in_)
                break

    if expected_cols is not None:
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[expected_cols]

    prob = float(model.predict_proba(df)[0, 1])
    pred = int(prob >= DECISION_THRESHOLD)

    return {
        "churn_prediction":  pred,
        "churn_probability": round(prob, 4),
        "churn_label":       "Churn" if pred == 1 else "No Churn",
        "risk_level":        classify_risk(prob),
    }


def preprocess_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Vectorized preprocessing of an entire DataFrame matching trained model features."""
    scaler, encoders, config = load_preprocessing_artifacts()
    df = df_raw.copy()

    # Binary columns
    for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        if col in df.columns:
            df[col] = df[col].astype(str).map(lambda x: 1 if x.strip().lower() in ["yes", "1", "true"] else 0)
        else:
            df[col] = 0

    # Service columns
    for col in SERVICE_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).map(lambda x: 0 if x.strip().lower() in ["no", "no internet service", "no phone service", "0", "false"] else 1)
        else:
            df[col] = 0

    # Contract
    if "Contract" in df.columns:
        df["Contract"] = df["Contract"].astype(str).map(lambda x: 2 if "two" in x.lower() else (1 if "one" in x.lower() else 0))
    else:
        df["Contract"] = 0

    # Categoricals with encoders
    for col in ["gender", "InternetService", "PaymentMethod"]:
        if col in df.columns:
            encoder = encoders.get(col)
            if encoder is not None and hasattr(encoder, "classes_"):
                known = set(encoder.classes_)
                df[col] = df[col].astype(str).map(lambda x: int(encoder.transform([x])[0]) if x in known else 0)
            else:
                fallback = {"gender": GENDER_MAP, "InternetService": INTERNET_MAP, "PaymentMethod": PAYMENT_MAP}[col]
                df[col] = df[col].astype(str).map(lambda x: fallback.get(x, 0))
        else:
            df[col] = 0

    # Numerics
    for col in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0

    # Match the training-time winsorisation before deriving features
    df = _clip_to_training_bounds(df, config)

    # Feature engineering
    total_derived = df["MonthlyCharges"] * df["tenure"]
    df["TotalCharges"] = np.where(df["TotalCharges"] > 0, df["TotalCharges"], total_derived)
    df["ChargesPerTenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)
    service_count_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies",
    ]
    df["ServiceCount"] = df[[c for c in service_count_cols if c in df.columns]].sum(axis=1)
    df["IsHighValue"] = (df["MonthlyCharges"] >= config["monthly_charge_threshold"]).astype(int)
    df["HasLoyalty"] = (df["tenure"] > 24).astype(int)
    contract_risk = (2 - df["Contract"]) / 2
    charges_norm = (df["MonthlyCharges"] - config["monthly_charge_min"]) / (
        config["monthly_charge_max"] - config["monthly_charge_min"] + 1e-9
    )
    tenure_norm = 1 - (df["tenure"] - config["tenure_min"]) / (
        config["tenure_max"] - config["tenure_min"] + 1e-9
    )
    df["RiskScore"] = (0.4 * contract_risk + 0.3 * charges_norm + 0.3 * tenure_norm).round(4)

    scale_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    df[scale_cols] = scaler.transform(df[scale_cols])
    return df


def predict_batch(df_raw: pd.DataFrame, model=None, require_schema: bool = True) -> pd.DataFrame:
    """
    Predict churn for a batch of customers from a raw DataFrame.
    Returns standardized columns:
      - 'Churn Probability' (0.0 to 100.0)
      - 'Churn Prediction'  ('Will Churn' / 'Will Stay')
      - 'Risk Level'         ('🔴 High' / '🟡 Medium' / '🟢 Low')
    Plus lowercase backward-compatible columns:
      - 'churn_probability' (0.0 to 1.0)
      - 'churn_prediction'  (0 or 1)
      - 'risk_level'         ('High' / 'Medium' / 'Low')

    Raises ValueError when ``require_schema`` is True and the frame is missing
    the Telco columns this model was trained on. Scoring an incompatible schema
    would fill every feature with 0 and return identical probabilities for all
    rows, so callers must route non-Telco data to the generic pipeline instead.
    """
    if require_schema and not is_telco_schema(df_raw):
        raise ValueError(
            "Dataset does not match the Telco churn model schema. "
            "Missing one or more of: tenure, MonthlyCharges, TotalCharges, "
            "Contract, InternetService, PaymentMethod. "
            "Use src.generic_pipeline for arbitrary schemas."
        )

    if model is None:
        model = load_model("best_model")

    try:
        X = preprocess_dataframe(df_raw)

        # ── Drop known non-feature columns that preprocess_dataframe passes through
        _drop = [c for c in X.columns if c.lower() in
                 {"churn", "customerid", "customer_id", "custid", "accountid",
                  "churn prediction", "churn_prediction", "churn probability",
                  "churn_probability", "risk level", "risk_level"}]
        if _drop:
            X = X.drop(columns=_drop, errors="ignore")

        # ── Align columns to what the model expects
        expected = None
        if hasattr(model, "feature_names_in_"):
            expected = list(model.feature_names_in_)
        elif hasattr(model, "estimators_") and model.estimators_:
            # VotingClassifier: grab feature names from first sub-estimator
            for _est in model.estimators_:
                if hasattr(_est, "feature_names_in_"):
                    expected = list(_est.feature_names_in_)
                    break

        if expected is not None:
            for c in expected:
                if c not in X.columns:
                    X[c] = 0
            X = X[expected]

        probs = model.predict_proba(X)[:, 1]
        # Apply the app-wide decision threshold instead of the estimator's
        # implicit 0.50 so batch, single and evaluation paths agree.
        preds = (probs >= DECISION_THRESHOLD).astype(int)
    except Exception as e:
        logger.warning(f"Vectorized batch prediction fallback: {e}")
        results = [predict_single(r.to_dict(), model) for _, r in df_raw.iterrows()]
        probs = np.array([r["churn_probability"] for r in results])
        preds = np.array([r["churn_prediction"] for r in results])

    out = df_raw.copy()
    out["Churn Probability"] = (probs * 100).round(1)
    out["Churn Prediction"]  = np.where(preds == 1, "Will Churn", "Will Stay")
    out["Risk Level"]        = [classify_risk_emoji(p) for p in probs]

    out["churn_probability"] = probs.round(4)
    out["churn_prediction"]  = preds.astype(int)
    out["risk_level"]        = [classify_risk(p) for p in probs]
    return out


if __name__ == "__main__":
    # Example prediction
    sample_customer = {
        "gender":          "Female",
        "SeniorCitizen":   0,
        "Partner":         "Yes",
        "Dependents":      "No",
        "tenure":          5,
        "PhoneService":    "Yes",
        "MultipleLines":   "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity":  "No",
        "OnlineBackup":    "No",
        "DeviceProtection":"No",
        "TechSupport":     "No",
        "StreamingTV":     "Yes",
        "StreamingMovies": "Yes",
        "Contract":        "Month-to-month",
        "PaperlessBilling":"Yes",
        "PaymentMethod":   "Electronic check",
        "MonthlyCharges":  90.45,
        "TotalCharges":    452.25,
    }

    result = predict_single(sample_customer)
    print("\nPrediction Result:")
    for k, v in result.items():
        print(f"  {k}: {v}")
