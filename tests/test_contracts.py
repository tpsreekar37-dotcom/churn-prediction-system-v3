"""
test_contracts.py
-----------------
Regression tests for invariants that previously broke silently:

1. The Telco model must refuse non-Telco schemas instead of returning a
   constant probability for every row.
2. Risk banding must be identical across modules.
3. Inference artifacts must be loadable and produce varied scores.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils import (  # noqa: E402
    classify_risk, classify_risk_emoji, is_telco_schema,
    RISK_HIGH, RISK_MEDIUM,
)
from src.predict import predict_batch, predict_single  # noqa: E402

DATA_DIR = os.path.join(ROOT, "data")
TELCO_SAMPLE = os.path.join(DATA_DIR, "sample_test_customers.csv")
UNIVERSAL_SAMPLE = os.path.join(DATA_DIR, "sample_universal_churn_test.csv")


# ── Risk banding ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("probability, expected", [
    (0.00, "Low"),
    (RISK_MEDIUM - 0.001, "Low"),
    (RISK_MEDIUM, "Medium"),
    (RISK_HIGH - 0.001, "Medium"),
    (RISK_HIGH, "High"),
    (1.00, "High"),
])
def test_risk_banding_is_consistent(probability, expected):
    assert classify_risk(probability) == expected
    assert expected in classify_risk_emoji(probability)


def test_risk_banding_handles_bad_input():
    assert classify_risk(None) == "Low"
    assert classify_risk(float("nan")) == "Low"


# ── Schema guard ──────────────────────────────────────────────────────────────

def test_telco_schema_detected():
    df = pd.read_csv(TELCO_SAMPLE)
    assert is_telco_schema(df) is True


def test_universal_schema_rejected():
    df = pd.read_csv(UNIVERSAL_SAMPLE)
    assert is_telco_schema(df) is False


def test_predict_batch_refuses_foreign_schema():
    df = pd.read_csv(UNIVERSAL_SAMPLE)
    with pytest.raises(ValueError, match="does not match the Telco"):
        predict_batch(df)


def test_predict_batch_allows_explicit_override():
    """require_schema=False keeps the old behaviour available for callers that
    genuinely want the model's fallback path."""
    df = pd.read_csv(UNIVERSAL_SAMPLE)
    out = predict_batch(df, require_schema=False)
    assert len(out) == len(df)


# ── Inference quality ─────────────────────────────────────────────────────────

def test_batch_predictions_are_not_constant():
    df = pd.read_csv(TELCO_SAMPLE)
    out = predict_batch(df)
    probs = out["churn_probability"]
    assert probs.nunique() > 1, "Model returned the same probability for every row"
    assert probs.between(0, 1).all()
    assert set(out["risk_level"]).issubset({"High", "Medium", "Low"})


def test_single_prediction_contract():
    customer = {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "tenure": 5, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No",
        "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "Yes", "StreamingMovies": "Yes",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 90.45, "TotalCharges": 452.25,
    }
    result = predict_single(customer)
    assert set(result) == {
        "churn_prediction", "churn_probability", "churn_label", "risk_level"
    }
    assert result["risk_level"] == classify_risk(result["churn_probability"])
    assert (result["churn_label"] == "Churn") == (result["churn_prediction"] == 1)


def test_out_of_range_charges_are_clipped():
    """Inference must apply the same winsorisation as training, so absurd inputs
    are capped at the training maximum instead of being extrapolated."""
    import json

    from src.utils import MODELS_DIR

    cfg_path = os.path.join(MODELS_DIR, "feature_config.json")
    if not os.path.exists(cfg_path):
        pytest.skip("feature_config.json not generated yet (run `python main.py`).")
    with open(cfg_path, encoding="utf-8") as fh:
        bounds = (json.load(fh).get("outlier_bounds") or {})
    if "MonthlyCharges" not in bounds:
        pytest.skip("Training bounds are empty — retrain with `python main.py`.")

    hi_mc = bounds["MonthlyCharges"][1]
    hi_tc = bounds["TotalCharges"][1]

    base = {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "tenure": 10, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "One year", "PaperlessBilling": "No",
        "PaymentMethod": "Mailed check", "MonthlyCharges": 50.0, "TotalCharges": 500.0,
    }
    at_max  = dict(base, MonthlyCharges=hi_mc, TotalCharges=hi_tc)
    extreme = dict(base, MonthlyCharges=1_000_000.0, TotalCharges=1_000_000.0)

    capped = predict_single(at_max)["churn_probability"]
    clipped = predict_single(extreme)["churn_probability"]

    assert 0.0 <= clipped <= 1.0
    # An absurd charge must score exactly like the training maximum, proving the
    # value was winsorised rather than fed to the model raw.
    assert abs(clipped - capped) < 1e-9
