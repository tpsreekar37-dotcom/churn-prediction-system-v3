# 📘 User Manual — Business Analytics System: Customer Churn Prediction

---

## 💻 1. System Requirements

- **Python**: 3.10 – 3.12
- **Pip**: included with Python
- **RAM**: 4 GB minimum, 8 GB recommended
- **Browser**: Chrome, Edge, Firefox, Safari or Brave

---

## 📦 2. Dependencies

All dependencies are listed in `requirements.txt`:

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `pandas`, `numpy` | Data manipulation |
| `scikit-learn` | Models, preprocessing, metrics |
| `xgboost` | Gradient-boosted benchmark model |
| `imbalanced-learn` | SMOTE class balancing |
| `plotly` | Interactive charts |
| `matplotlib`, `seaborn` | Static report figures |
| `openpyxl` | Excel (`.xlsx`) upload support |
| `joblib` | Model serialisation |
| `sqlalchemy`, `scipy` | SQL helpers and statistics |

> ⚠️ Loading a `.pkl` trained under a different scikit-learn version can change
> predictions silently. `models/model_metadata.json` records the version used at
> training time; `python main.py` regenerates artifacts with your installed
> version. Re-run it after upgrading scikit-learn.

---

## 🚀 3. Installation & Launch

### Step 1 — Open a terminal in the project folder

```cmd
cd churn_predicting
```

### Step 2 — (Recommended) create a virtual environment

```cmd
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
```

### Step 3 — Install dependencies

```cmd
pip install -r requirements.txt
```

### Step 4 — Launch the dashboard

```cmd
streamlit run dashboard/app.py
```

### Step 5 — Open the app

👉 **http://localhost:8501**

### Optional — Rebuild all artefacts from raw data

```cmd
python main.py
```

This reruns preprocessing, feature engineering, EDA, training and the SQL load,
and rewrites everything in `models/`, `data/processed/` and `reports/figures/`.

---

## 📊 4. Dashboard Pages

The app has **six pages** in the left sidebar.

### 4.1 Dataset Import & Model Selection
Upload a `.csv`/`.xlsx`, load the benchmark, validate the file, and choose which
model scores the portfolio. The per-model cards show **Accuracy** as the
headline with `F1 | AUC` beneath. These are the same numbers shown on every
other page.

### 4.2 Executive Overview
Portfolio-level KPIs: accounts scored, predicted churn rate, value at risk,
high-risk headcount, and the churn distribution chart.

### 4.3 Business Revenue & Financial Risk
- Value at risk, annualised exposure (monthly columns only), ARPU, and
  churner-vs-portfolio ARPU.
- Churn by contract type, payment method and tenure cohort.
- **Retention Campaign ROI Simulator**: set the intervention success rate and
  incentive cost; the app projects value preserved, campaign cost, net gain and
  ROI %.

**Which column is used?** The app looks for a monetary column in this order:
`MonthlyCharges` / `monthly` / `mrr` / `charges` (treated as monthly), then
`balance` / `account balance` / `amount` / `revenue` / `ltv` / `salary` /
`income` (treated as a per-customer stock value). Monthly columns are
annualised; stock columns are not. If nothing is found the tiles read **N/A** —
the app never invents a dollar figure.

### 4.4 Model Comparison & Evaluation
Train the generic pipeline on the uploaded file (when it is not Telco schema) and
score each candidate model on the same held-out split. Shows the comparison
chart, per-model KPI cards and the confusion matrix / ROC curve for the
evaluation run.

### 4.5 Model Testing & Evaluation
- Trained model benchmarks (canonical table).
- Upload labelled test data, or score the built-in benchmark.
- Accuracy, precision, recall, F1 and the confusion matrix.
- Side-by-side actual-vs-predicted comparison.
- **Epoch training telemetry**: trains a small neural network one epoch at a
  time (`warm_start=True`, `max_iter=1`) and charts real train/validation
  accuracy and log-loss per epoch. These are measurements from an actual fit,
  not a simulated curve.

### 4.6 Customer Retention Strategy
Segment the scored portfolio and generate prioritised retention actions with
estimated value impact. Money labels follow the detected column, so a bank
`balance` dataset shows "Account Balance at Risk", not "MRR".

---

## 🎚️ 5. Scores, Thresholds and Why They Match Everywhere

- A single threshold, `DECISION_THRESHOLD = 0.50`, converts probability to
  "Will Churn"/"Will Stay" in training, batch scoring and single-customer
  scoring.
- The Youden-J threshold saved during training is informational only and is not
  used for predictions.
- All per-model metrics come from one canonical table in
  `src/model_registry.py`. Because every page reads that table, a model's
  accuracy/F1/AUC cannot differ between pages or sessions.
- For the Telco benchmark the table is the saved holdout result
  (`models/model_results.json`). For a non-Telco upload it is that upload's own
  holdout split, which is deterministic for a given file and candidate set.
- The live "Ground-Truth" section on Model Testing & Evaluation re-scores the
  rows you supply on screen. It is captioned with the canonical benchmark value
  and the delta, because a different set of rows legitimately gives a slightly
  different score.

---

## ❓ 6. Troubleshooting

**Port 8501 already in use**
```cmd
streamlit run dashboard/app.py --server.port 8502
```

**`ModuleNotFoundError`**
Install dependencies into the same environment that runs `streamlit`:
```cmd
pip install -r requirements.txt
```

**`InconsistentVersionWarning` when loading a model**
The `.pkl` was produced by a different scikit-learn version. Regenerate:
```cmd
python main.py
```

**"Dataset does not match the Telco churn model schema"**
The upload lacks required columns. Either supply the Telco columns or include a
binary churn column so the generic pipeline can train on it.

**Predictions look identical for every row**
This was a bug in earlier versions: non-Telco data was silently scored by the
Telco model with all features defaulted to 0. Fixed — the app now trains a
schema-aware pipeline on the upload and refuses to score incompatible data.

**Uploading a bank-style dataset fails**
Earlier builds raised `AttributeError: 'NoneType' object has no attribute
'transform'` because a binary 0/1 column was sent through a categorical encoder
that did not exist. Fixed in `src/generic_pipeline.py`; binary columns are now
recognised and passed through untouched.

**"Revenue at risk" shows N/A**
The dataset has no recognisable monetary column. Add one named like
`MonthlyCharges`, `charges`, `balance`, `amount`, `revenue`, `ltv`, `salary` or
`income`. The app reports N/A on purpose rather than inventing a number.

**"Revenue at risk" is much larger than expected on a bank dataset**
A bank file's `balance` is a per-customer stock value, so it is summed as-is and
is not annualised. Only monthly columns are multiplied by 12.

**A model shows a different accuracy on two pages**
This was fixed by routing every page through the canonical metric table. If you
still see a mismatch, check the caption under the table: the *Live Ground-Truth*
panel re-scores the rows you uploaded, so it can differ from the saved benchmark
by a few points — the caption states the canonical figure and the delta. Run
`python main.py` if `models/model_results.json` is missing or stale.

---

## 📜 Authors & License

Built with Python, scikit-learn, XGBoost and Streamlit for customer churn and
revenue-risk analytics. MIT License.
