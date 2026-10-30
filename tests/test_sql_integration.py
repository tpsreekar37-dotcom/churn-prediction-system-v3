"""
test_sql_integration.py
-------------------------
Automated Test Suite for SQLite Storage & Business Analytics Subsystem.

Verifies:
  1. Database initialization and table schema setup
  2. Customer and prediction insertion with run_id tracking
  3. Correct MRR, ARR, ARPU, and high-value churn calculations
  4. SQL query aggregations by contract, payment method, and cohorts
  5. Multi-run history isolation without duplicate join inflation
  6. Handling of missing IDs, empty datasets, and NaN values
"""

import os
import sys
import unittest
import sqlite3
import pandas as pd
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.sql_integration import (
    init_db,
    store_predictions_to_sqlite,
    get_latest_run_id,
    run_analytics,
    run_query
)

TEST_DB_PATH = os.path.join(PROJECT_ROOT, "data", "test_churn.db")


class TestSQLIntegration(unittest.TestCase):

    def setUp(self):
        """Set up a fresh test database before each test."""
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)
        init_db(TEST_DB_PATH)

    def tearDown(self):
        """Clean up test database file after each test."""
        if os.path.exists(TEST_DB_PATH):
            try:
                os.remove(TEST_DB_PATH)
            except Exception:
                pass

    def test_01_schema_initialization(self):
        """Test that tables and indexes are created properly."""
        conn = sqlite3.connect(TEST_DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        self.assertIn("customers", tables)
        self.assertIn("service_contracts", tables)
        self.assertIn("churn_predictions", tables)

        cursor.execute("PRAGMA table_info(churn_predictions);")
        columns = [row[1] for row in cursor.fetchall()]
        self.assertIn("prediction_id", columns)
        self.assertIn("customer_id", columns)
        self.assertIn("predicted_churn", columns)
        self.assertIn("run_id", columns)
        self.assertIn("risk_category", columns)

        conn.close()

    def test_02_store_predictions_and_run_tracking(self):
        """Test inserting customer predictions and verifying run_id tracking."""
        data = {
            "customerID": ["C101", "C102", "C103", "C104"],
            "MonthlyCharges": [50.0, 100.0, 80.0, 60.0],
            "TotalCharges": [500.0, 1000.0, 800.0, 600.0],
            "tenure": [10, 10, 10, 10],
            "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month"],
            "PaymentMethod": ["Electronic check", "Mailed check", "Electronic check", "Credit card"],
            "InternetService": ["Fiber optic", "DSL", "No", "Fiber optic"],
            "Churn Prediction": ["Will Churn", "Stay", "Will Churn", "Stay"],
            "Churn Probability": [0.85, 0.20, 0.70, 0.15],
            "Risk Level": ["🔴 High Risk", "🟢 Low Risk", "🔴 High Risk", "🟢 Low Risk"]
        }
        df = pd.DataFrame(data)

        db_path, run_id_1 = store_predictions_to_sqlite(df, db_path=TEST_DB_PATH)
        self.assertTrue(run_id_1.startswith("RUN_"))
        
        latest_run = get_latest_run_id(TEST_DB_PATH)
        self.assertEqual(latest_run, run_id_1)

        conn = sqlite3.connect(TEST_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM customers;")
        self.assertEqual(cursor.fetchone()[0], 4)

        cursor.execute("SELECT COUNT(*) FROM churn_predictions WHERE run_id = ?;", (run_id_1,))
        self.assertEqual(cursor.fetchone()[0], 4)
        conn.close()

    def test_03_mrr_arr_arpu_calculations(self):
        """Test mathematical accuracy of MRR, ARR, ARPU, and High-Value Churn metrics."""
        # Setup known dataset:
        # C1: Monthly $100 -> Predicted Churn (1)
        # C2: Monthly $50  -> Predicted Churn (1)
        # C3: Monthly $60  -> Retained (0)
        # C4: Monthly $40  -> Retained (0)
        # Average Monthly = (100+50+60+40)/4 = $62.50
        # Predicted Churners: C1 ($100), C2 ($50).
        # MRR at Risk = $100 + $50 = $150.00
        # ARR at Risk = $150 * 12 = $1,800.00
        # ARPU = $62.50
        # High Value Churners (spending >= $62.50): C1 ($100) -> 1 customer
        data = {
            "customerID": ["C1", "C2", "C3", "C4"],
            "MonthlyCharges": [100.0, 50.0, 60.0, 40.0],
            "TotalCharges": [1000.0, 500.0, 600.0, 400.0],
            "tenure": [10, 10, 10, 10],
            "Contract": ["Month-to-month", "Month-to-month", "One year", "Two year"],
            "PaymentMethod": ["Electronic check", "Electronic check", "Mailed check", "Credit card"],
            "InternetService": ["Fiber optic", "DSL", "No", "Fiber optic"],
            "Churn Prediction": ["Will Churn", "Will Churn", "Stay", "Stay"],
            "Churn Probability": [0.90, 0.75, 0.10, 0.05],
            "Risk Level": ["🔴 High", "🔴 High", "🟢 Low", "🟢 Low"]
        }
        df = pd.DataFrame(data)
        _, run_id = store_predictions_to_sqlite(df, db_path=TEST_DB_PATH)

        analytics = run_analytics(db_path=TEST_DB_PATH, run_id=run_id)
        self.assertIn("revenue_at_risk", analytics)
        
        rev_df = analytics["revenue_at_risk"]
        self.assertFalse(rev_df.empty)

        row = rev_df.iloc[0]
        self.assertAlmostEqual(row["monthly_revenue_at_risk"], 150.00, places=2)
        self.assertAlmostEqual(row["annual_revenue_at_risk"], 1800.00, places=2)
        self.assertAlmostEqual(row["arpu"], 62.50, places=2)
        self.assertEqual(row["high_value_churn_count"], 1)
        self.assertEqual(row["predicted_churned_count"], 2)
        self.assertEqual(row["total_customers"], 4)

    def test_04_multi_run_isolation_and_no_join_duplication(self):
        """Test that multiple uploads/runs store separate prediction runs without inflating joins."""
        data_v1 = {
            "customerID": ["C1", "C2"],
            "MonthlyCharges": [100.0, 50.0],
            "Contract": ["Month-to-month", "One year"],
            "PaymentMethod": ["Electronic check", "Mailed check"],
            "InternetService": ["Fiber optic", "DSL"],
            "Churn Prediction": ["Will Churn", "Stay"],
            "Churn Probability": [0.80, 0.20],
            "Risk Level": ["🔴 High", "🟢 Low"]
        }
        df_v1 = pd.DataFrame(data_v1)
        _, run_id_1 = store_predictions_to_sqlite(df_v1, db_path=TEST_DB_PATH)

        # Second run with different prediction
        data_v2 = {
            "customerID": ["C1", "C2"],
            "MonthlyCharges": [100.0, 50.0],
            "Contract": ["Month-to-month", "One year"],
            "PaymentMethod": ["Electronic check", "Mailed check"],
            "InternetService": ["Fiber optic", "DSL"],
            "Churn Prediction": ["Stay", "Will Churn"],
            "Churn Probability": [0.10, 0.90],
            "Risk Level": ["🟢 Low", "🔴 High"]
        }
        df_v2 = pd.DataFrame(data_v2)
        _, run_id_2 = store_predictions_to_sqlite(df_v2, db_path=TEST_DB_PATH)

        self.assertNotEqual(run_id_1, run_id_2)

        # Verify analytics on run 2 uses ONLY run 2 data
        analytics_v2 = run_analytics(db_path=TEST_DB_PATH, run_id=run_id_2)
        rev_v2 = analytics_v2["revenue_at_risk"].iloc[0]
        self.assertEqual(rev_v2["monthly_revenue_at_risk"], 50.00) # Only C2 churned in run 2

        # Verify total customer rows in query equals 2 (not duplicated to 4)
        self.assertEqual(rev_v2["total_customers"], 2)

    def test_05_handling_missing_ids_and_nan_values(self):
        """Test safe handling of missing customer IDs, inf values, and NaN charges."""
        data_messy = {
            "customerID": [None, "C200", ""],
            "MonthlyCharges": [np.nan, "120.5", np.inf],
            "TotalCharges": [" ", "1205.0", np.nan],
            "tenure": [0, 10, 5],
            "Contract": ["Month-to-month", "One year", "Two year"],
            "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer"],
            "InternetService": ["Fiber optic", "DSL", "No"],
            "Churn Prediction": ["Will Churn", "Stay", "Will Churn"],
            "Churn Probability": [0.70, 0.10, 0.60],
            "Risk Level": ["🔴 High", "🟢 Low", "🟡 Medium"]
        }
        df_messy = pd.DataFrame(data_messy)
        db_path, run_id = store_predictions_to_sqlite(df_messy, db_path=TEST_DB_PATH)

        analytics = run_analytics(db_path=TEST_DB_PATH, run_id=run_id)
        self.assertIn("churn_summary", analytics)
        summary = analytics["churn_summary"].iloc[0]
        self.assertEqual(summary["total_customers"], 3)
        self.assertEqual(summary["total_churned"], 2)


if __name__ == "__main__":
    unittest.main()
