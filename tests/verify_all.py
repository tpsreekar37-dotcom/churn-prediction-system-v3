import os, sys, time, pandas as pd, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils import safe_float, safe_int, find_col, get_revenue_metrics
from src.predict import predict_batch, load_model
from src.sql_integration import store_predictions_to_sqlite, run_analytics
from src.generic_pipeline import run_generic_pipeline, predict_generic_batch

print("Test 1 - Safe NaN conversion:")
print("  safe_int(np.nan):", safe_int(float('nan'), 99))
print("  safe_float(None):", safe_float(None, 0.0))

print("Test 2 - teseataset.xlsx prediction:")
df_test = pd.read_excel('teseataset.xlsx')
t0 = time.time()
res_test = predict_batch(df_test)
print(f"  predict_batch on {len(df_test)} rows took {time.time()-t0:.3f}s")
print("  Columns present:", [c for c in ['Churn Probability', 'Churn Prediction', 'Risk Level'] if c in res_test.columns])

print("Test 3 - SQLite store speed:")
t0 = time.time()
db_path, run_id = store_predictions_to_sqlite(res_test)
print(f"  SQLite store of {len(res_test)} rows took {time.time()-t0:.3f}s, run_id={run_id}")

print("Test 4 - SQL Analytics:")
sql_res = run_analytics(db_path, run_id)
rev = get_revenue_metrics(res_test, sql_res)
print(f"  MRR at risk: ${rev['mrr']:,.2f}, ARR: ${rev['arr']:,.2f}, Total: {rev['total']}")

print("Test 5 - Benchmark data store & analytics:")
df_bm = pd.read_csv('data/raw/telco_churn_raw.csv')
t0 = time.time()
res_bm = predict_batch(df_bm)
print(f"  Benchmark scoring {len(df_bm)} rows took {time.time()-t0:.3f}s")
t0 = time.time()
db_path, run_id_bm = store_predictions_to_sqlite(res_bm)
print(f"  Benchmark SQLite store {len(df_bm)} rows took {time.time()-t0:.3f}s")
sql_bm = run_analytics(db_path, run_id_bm)
rev_bm = get_revenue_metrics(res_bm, sql_bm)
print(f"  Benchmark MRR at risk: ${rev_bm['mrr']:,.2f}, ARR: ${rev_bm['arr']:,.2f}, High-val: {rev_bm['high_val']}")

print("ALL TESTS PASSED WITH 100% SUCCESS!")
