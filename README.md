# Business Analytics System: AI-Driven Customer Churn Prediction

An end-to-end machine learning system that predicts customer churn, quantifies
revenue exposure from whichever monetary column the dataset provides, and
simulates retention campaign ROI.

> **Accuracy of claims:** this project contains no LLM/AI-generated narrative
> feature. All insights are computed from the trained classifier and SQL
> aggregates. Earlier revisions of this README advertised a Gemini integration
> that was never shipped.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-orange?logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29-red?logo=streamlit)

---

## Business Overview

In subscription and telecom businesses, acquiring a new customer costs 5–7×
more than retaining an existing one. Customer churn directly erodes recurring
revenue and enterprise valuation.

### Capabilities

1. **Batch Dataset Upload & Prediction Engine**
   - Upload a CSV or Excel customer dataset.
   - Automatic target and feature column detection.
   - Per-customer churn probability and risk tier (High / Medium / Low).
   - 10% binned churn probability distribution chart.

2. **Business Revenue & Risk Telemetry**
   - **Value at risk** is computed from the scored rows using the first monetary
     column found, in this priority order: `MonthlyCharges` / `mrr` / `charges`
     (monthly), then `balance` / `amount` / `ltv` / `revenue` / `salary` /
     `income` (one-off or stock values).
   - Monthly columns get an **annualised exposure** figure (`× 12`). Stock
     columns such as a bank `balance` are **not** multiplied and are labelled
     with their own name ("Account Balance at Risk"), never "MRR".
   - If no monetary column exists, every revenue tile shows **N/A** instead of a
     fabricated `$0` or a hard-coded per-customer guess.
   - ARPU comparison between predicted churners and the whole portfolio.
   - **Retention Campaign ROI Simulator**: adjust intervention success rate and
     incentive cost to project value preserved, cost, net gain and ROI.

3. **SQL Analytics Subsystem**
   - SQLite schema for customers, contracts and prediction runs.
   - Run-isolated history via `run_id`, so re-scoring never mixes cohorts.
   - Parameterised queries for churn by contract, payment method, tenure cohort
     and revenue at risk.

4. **Model Evaluation**
   - Logistic Regression, Random Forest, XGBoost, SVM, LightGBM and a voting
     ensemble, compared on the same held-out split.
   - ROC curves, confusion matrix and feature importance.
   - **One canonical metrics table** (`src/model_registry.py`) feeds every page,
     so accuracy / precision / recall / F1 / ROC-AUC for a given model are the
     same numbers everywhere in the app and across sessions.
   - Real iterative training telemetry (an MLP trained epoch-by-epoch with
     `warm_start`), not a simulated curve.

---

## Quick Start

```cmd
cd churn_predicting
pip install -r requirements.txt
streamlit run dashboard/app.py
```

Open **http://localhost:8501**.

To regenerate every artefact from raw data (preprocessing, features, EDA,
training, figures, SQL load):

```cmd
python main.py
```

---

## Deploy (Render, or Railway)

The repo is deploy-ready: `render.yaml` (Render Blueprint), `Procfile`
(Railway/Heroku), `.python-version` (3.12) and `.streamlit/config.toml` are
included, and the trained models in `models/` are committed on purpose — the app
loads them at startup instead of retraining.

```cmd
git init
git add .
git commit -m "Churn prediction dashboard"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

### Free hosting

**Streamlit Community Cloud** is the easiest free option (no card, 1 GB RAM):
<https://share.streamlit.io> → **Create app** → repo
`tpsreekar37-dotcom/churn-prediction-system-v3`, branch `main`, main file
`dashboard/app.py`, Python **3.12**. It installs `requirements.txt` and gives you
a `*.streamlit.app` URL.

> **Python 3.12 is required, and it must be chosen *before* deploying.** The
> pickles need scikit-learn 1.3.2, and the pinned dependencies have no wheels
> above 3.12 — on 3.13/3.14 the build fails while compiling `pillow` / `pandas`.
> Community Cloud cannot change an app's Python version in place: delete the app
> and redeploy with **Advanced settings → Python version: 3.12**.

**Render** works too and uses the included `render.yaml`:
**New + → Blueprint →** pick the repo → **Apply**. Render sets the Python
version, build command, start command and health check for you (free plan:
512 MB, sleeps after 15 min).

### Notes

- `.streamlit/secrets.toml` is **not** committed. Add keys as host environment
  variables instead (`GEMINI_API_KEY=...`).
- `data/churn.db` is not committed; the app creates it on first write. Both
  Render and Railway have ephemeral filesystems, so prediction history resets on
  each deploy.
- `models/best_model.pkl` is ~80 MB. It is under GitHub's 100 MB per-file limit,
  but the first push uploads ~107 MB of model artefacts.
- Full step-by-step, including manual Render setup and troubleshooting, is in
  `DEPLOY.md`.

---

## Project Structure

```
churn_predicting/
│
├── data/
│   ├── raw/                    ← Raw IBM Telco dataset
│   ├── processed/              ← Cleaned train/test splits
│   └── churn.db                ← SQLite analytics store (generated, not committed)
│
├── dashboard/
│   ├── app.py                  ← Streamlit dashboard (all six pages)
│   └── style.css               ← Theme
│
├── src/
│   ├── preprocessing.py        ← Cleaning, split, encoding, scaling
│   ├── feature_engineering.py  ← Derived features + SMOTE
│   ├── train_model.py          ← Model training & evaluation
│   ├── predict.py              ← Inference engine
│   ├── generic_pipeline.py     ← Schema-agnostic pipeline for uploads
│   ├── model_registry.py       ← Canonical metric table + model selection
│   ├── sql_integration.py      ← SQLite persistence & analytics
│   ├── eda.py                  ← Exploratory figures
│   └── utils.py                ← Shared constants, risk bands, helpers
│
├── models/                     ← Trained artefacts (committed — the app loads them at startup)
│   ├── best_model.pkl          ← Calibrated best model
│   ├── model_metadata.json     ← Runtime/library versions + metrics
│   ├── model_results.json      ← Canonical per-model holdout metrics
│   ├── scaler.pkl / encoders.pkl
│   └── feature_config.json     ← Thresholds and outlier bounds from the training split
│
├── reports/figures/            ← EDA and evaluation charts
├── slide_exports/              ← Presentation slide images
├── sql/                        ← Reference analytics queries
├── scripts/legacy/             ← One-off report/deck generators (not used at runtime)
├── tests/                      ← pytest suite
├── Procfile / .python-version  ← Deploy entry point and runtime pin
├── requirements.txt
├── DEPLOY.md                   ← GitHub → Railway guide
└── README.md
```

---

## Testing with an upload

1. Launch the dashboard.
2. Open **Dataset Import & Model Selection**.
3. Upload a Telco-schema CSV, any other CSV with a churn column (for example a
   bank churn file with `balance` and `churn` columns), or click
   **Load Benchmark**.
4. Pick a model (or leave **Auto (Best)**) and click **Apply & Score Portfolio**.
5. Review risk scores, the probability distribution and the results table;
   download the scored CSV.

The dashboard routes automatically:

| Upload contains | Scoring path |
|---|---|
| `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `InternetService`, `PaymentMethod` | Pre-trained Telco models |
| Any other schema with a binary churn column | `generic_pipeline` trains and evaluates its own candidate models on the upload |
| Neither | Explicit error message — no placeholder predictions |

For a non-Telco upload the per-model metrics shown in the app are that upload's
own held-out split results. For the benchmark Telco dataset they are the saved
values in `models/model_results.json`. In both cases the same table is reused on
every page, so a model never shows two different accuracies.

---

## Model & Metrics

### Decision threshold

A single threshold — `DECISION_THRESHOLD = 0.50` in `src/utils.py` — turns a
probability into "Will Churn" / "Will Stay". Training-time evaluation, batch
scoring, single-customer scoring and the dashboards all use it, so the same
probability always produces the same label.

The Youden-J optimal threshold found during training is stored as
`youden_threshold_informational` in `models/model_metadata.json`. It is
**informational only** and is deliberately not used for scoring: mixing the two
was what produced the old "81.9% here, 73% there" discrepancy.

### One canonical metrics table

`src/model_registry.py` exposes `metric_rows()` and `canonical_metric_rows()`.
Every page calls one of them:

| Page | Source |
|---|---|
| Dataset Import & Model Selection | canonical rows |
| Model Comparison & Evaluation | canonical rows (one pipeline trained per upload, each candidate model scored separately) |
| Model Testing & Evaluation | canonical rows for the benchmark cards |
| Model Testing & Evaluation → Live Ground-Truth | live re-score, explicitly captioned with the canonical figure and the delta |

The holdout metrics are written to `models/model_results.json` after each
training run; `models/model_metadata.json` records the library versions that
produced the artifacts. Do not compare metrics across runs without checking
that file first — a scikit-learn version change can move results.

Known limitations:

- The 0.50 threshold favours recall, so the model over-predicts churn relative to
  the base rate. Risk bands are advisory.
- Probabilities are calibrated with `CalibratedClassifierCV` when that does not
  cost more than 0.005 ROC-AUC; otherwise the uncalibrated model is kept and
  `calibrated` is `false` in the metadata.
- The Telco model must not be used on non-Telco data; the schema guard prevents
  this, but it means "universal" scoring always retrains on the upload.
- No drift monitoring, and no scheduled retraining.
- Quantile winsorisation bounds are learned on the training split and stored in
  `models/feature_config.json`. Inference clips to those bounds, so an absurd
  input (e.g. a `MonthlyCharges` of 1,000,000) scores exactly like the largest
  value the model ever saw.

---

## License

MIT License — built for professional enterprise demonstration.
