"""
test_pipeline_e2e.py
--------------------
End-to-end verification of model inference, generic pipeline, and SQL analytics.
"""
import os
import sys
try:
    import pytest
except ImportError:
    pytest = None
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from src.predict import predict_single, predict_batch, load_model
from src.generic_pipeline import run_generic_pipeline, predict_generic_batch
from src.sql_integration import store_predictions_to_sqlite, run_analytics, get_latest_run_id

def test_single_customer_prediction():
    cust = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.0,
        "TotalCharges": 1020.0,
    }
    res = predict_single(cust)
    assert "churn_prediction" in res
    assert "churn_probability" in res
    assert "risk_level" in res
    assert 0.0 <= res["churn_probability"] <= 1.0
    assert res["risk_level"] in ["High", "Medium", "Low"]

def test_generic_pipeline_flow():
    df_raw = pd.read_csv(os.path.join("data", "sample_test_customers.csv"))
    target_col = "Churn"
    id_col = "customerID" if "customerID" in df_raw.columns else None

    pipeline_result = run_generic_pipeline(df_raw, target_col, id_col)
    assert pipeline_result is not None
    assert "best_model" in pipeline_result

    pred_df = predict_generic_batch(df_raw, pipeline_result, target_col, id_col)
    assert "Churn Probability" in pred_df.columns
    assert "Churn Prediction" in pred_df.columns
    assert "Risk Level" in pred_df.columns
    assert len(pred_df) == len(df_raw)

def test_sql_storage_and_telemetry():
    # Use a scratch database: writing to data/churn.db would pollute the live
    # dashboard's run history every time the test suite runs.
    db_path = os.path.join(PROJECT_ROOT, "data", "test_churn.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    df_raw = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "sample_test_customers.csv"))
    pipeline_result = run_generic_pipeline(df_raw, "Churn", "customerID")
    pred_df = predict_generic_batch(df_raw, pipeline_result, "Churn", "customerID")

    returned_path, run_id = store_predictions_to_sqlite(pred_df, db_path=db_path)
    assert returned_path == db_path
    assert os.path.exists(returned_path)
    assert run_id is not None

    analytics = run_analytics(db_path=db_path, run_id=run_id)
    assert "revenue_at_risk" in analytics
    assert "churn_by_contract" in analytics
    assert "high_risk_customers" in analytics

    os.remove(db_path)
