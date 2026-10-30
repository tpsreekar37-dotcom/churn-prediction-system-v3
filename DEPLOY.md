# Deployment Guide — Free Hosting

The project ships ready to deploy. Nothing is trained or built on the server:
`models/` contains the trained artefacts, so the app starts scoring as soon as
the container boots. There are **no API keys and no secrets** to configure.

Two free paths, both no-credit-card:

| Host | RAM | Sleeps when idle | Setup |
|---|---|---|---|
| **Streamlit Community Cloud** (recommended) | 1 GB | yes | 2 minutes, click-through |
| **Render** (Blueprint) | 512 MB | yes (after 15 min) | 5 minutes, uses `render.yaml` |

---

## 1. What is already prepared

| File | Purpose |
|---|---|
| `render.yaml` | Render Blueprint: Python 3.12.5, build + start command, health check, auto-deploy |
| `Procfile` | Same start command, for Railway / Heroku / any process manager |
| `.python-version` | Pins Python 3.12 (the version that produced the model pickles) |
| `requirements.txt` | All runtime dependencies, pinned |
| `.streamlit/config.toml` | `headless = true`, 200 MB upload cap, file watcher off |
| `.gitignore` | Ignores secrets and the SQLite DB — **keeps** `models/*.pkl` |

The app never reads `st.secrets` or environment variables, so a host only needs
to run `streamlit run dashboard/app.py`.

---

## 2. Streamlit Community Cloud (free, no card) — recommended

This is the native home for a Streamlit app: it clones the repo, installs
`requirements.txt`, runs the app and gives you a public `*.streamlit.app` URL.
It also redeploys automatically on every push.

1. Sign in at <https://share.streamlit.io> with the GitHub account that owns the
   repo (`tpsreekar37-dotcom`).
2. Click **Create app** (or **New app**).
3. Choose **Deploy a public app from GitHub** and fill in:

   | Field | Value |
   |---|---|
   | Repository | `tpsreekar37-dotcom/churn-prediction-system-v3` |
   | Branch | `main` |
   | Main file path | `dashboard/app.py` |
   | App URL | pick something like `churn-prediction-system` |

4. Open **Advanced settings → Python version** and select **3.12** (matches the
   pickles in `models/`).
5. Click **Deploy**. The first build takes 3–6 minutes: it clones ~130 MB
   (the model artefacts) and installs the pinned dependencies.
6. Your app appears at `https://<app-url>.streamlit.app`.

No secrets to add, no start command to type, no port to set — Streamlit Cloud
handles all of that.

### Notes for this app

- **1 GB RAM.** The dashboard itself loads one 14 MB model and runs fine. The
  *Model Comparison & Evaluation* page trains five models on the uploaded file
  and is the heaviest page; on a very large CSV it can hit the limit.
- **Sleeps when idle.** The first visit after a long pause takes 30–60 s to
  wake. Open the URL a minute before a demo.
- **Ephemeral disk.** `data/churn.db` is recreated on each restart; nothing
  user-facing depends on it.
- **Redeploys on push** to `main`, and the app's own menu (⋮ → **Reboot app**)
  restarts it if needed.

---

## 3. Render (free, no card)

### Option A — Blueprint (uses `render.yaml`)

1. <https://dashboard.render.com> → **New +** → **Blueprint**.
2. Connect GitHub and pick `churn-prediction-system-v3`.
3. Render reads `render.yaml` and shows one web service,
   `churn-prediction-system`, on the **Free** plan.
4. Click **Apply**, wait for `Your service is live`, open the service URL.

`render.yaml` already sets `PYTHON_VERSION=3.12.5`, the build command, the start
command, the health check path and `autoDeploy: true`.

### Option B — Manual

1. **New +** → **Web Service** → pick the repository.
2. Settings:

   | Field | Value |
   |---|---|
   | Language / Runtime | **Python 3** |
   | Build Command | `pip install --upgrade pip && pip install -r requirements.txt` |
   | Start Command | `streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true --server.fileWatcherType none` |
   | Health Check Path | `/_stcore/health` |
   | Instance Type | Free |

3. **Advanced → Add Environment Variable**: `PYTHON_VERSION` = `3.12.5`.
4. **Create Web Service**.

### Expected build log

```
==> Installing Python version 3.12.5
==> Running build command 'pip install --upgrade pip && pip install -r requirements.txt'
Successfully installed pandas-2.1.4 scikit-learn-1.3.2 streamlit-1.29.0 ...
==> Starting service with 'streamlit run dashboard/app.py --server.port $PORT ...'

  You can now view your Streamlit app in your browser.
  URL: http://0.0.0.0:10000
==> Your service is live 🎉
```

### Render free-plan limits

| Limit | Effect on this app |
|---|---|
| **512 MB RAM** | Startup is fine; the training page is the risk (see §2 notes). |
| **Sleeps after 15 min idle** | First visit takes 30–60 s to wake. |
| **Ephemeral disk** | `data/churn.db` resets on deploy/restart. |

---

## 4. Pushing the project to GitHub (if you ever start over)

```cmd
git init
git add .
git commit -m "Churn prediction dashboard"
git branch -M main
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```

Confirm on github.com that these arrived:

- `render.yaml`, `Procfile`, `.python-version`
- `models/best_model.pkl` (~80 MB), `models/voting_ensemble.pkl` (~14 MB),
  `models/model_results.json`
- `data/raw/telco_churn_raw.csv`

If `models/*.pkl` is missing, an old `.gitignore` is excluding them. Restore the
one that ships with this project, or force-add:

```cmd
git add -f models/*.pkl
git commit -m "Add trained models"
git push
```

---

## 5. Troubleshooting

**Build fails on a dependency**
The pins are a tested set (pandas 2.1.4, scikit-learn 1.3.2, xgboost 2.0.3,
streamlit 1.29.0). On Render, make sure `PYTHON_VERSION=3.12.5` is set on the
service; on Streamlit Cloud, pick Python 3.12 in Advanced settings.

**`FileNotFoundError: models/...` in the logs**
The pickles were not committed — see §4.

**Page loads but stays on "RUNNING…"**
The host's proxy is blocking the Streamlit websocket. Uncomment the last two
lines of `.streamlit/config.toml` (`enableCORS`, `enableXsrfProtection`) and
redeploy.

**Render: service marked unhealthy / "no open ports detected"**
Check the health path is `/_stcore/health` and that the start command contains
`--server.port $PORT --server.address 0.0.0.0`.

**Render: `Killed` or the app dies on the training page**
Out of memory on the 512 MB free plan. Use the Streamlit Cloud path (1 GB) or a
paid instance for the demo.

**Numbers differ between pages**
They should not. Every page reads the canonical table in
`src/model_registry.py`; see §5 of `USER_MANUAL.md`.

---

## 6. Updating the deployed app

```cmd
git add -A
git commit -m "what changed"
git push
```

Both hosts redeploy automatically on a push to `main`.

---

## 7. Other hosts

- **Hugging Face Spaces** — free CPU tier with 16 GB RAM, no card. Requires a
  `README.md` at the repo root whose YAML header sets `sdk: streamlit` and
  `app_file: dashboard/app.py`; use a dedicated repo for it so this README stays
  untouched.
- **Railway** — `Procfile` is included; Railway injects `$PORT` the same way.
  Free credit is limited, so it is not a long-term free option.
- **Docker** — any `python:3.12-slim` image works:
  `CMD streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
