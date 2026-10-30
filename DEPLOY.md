# Deployment Guide — Hosting on Render

This project ships ready to deploy. Nothing is trained or built on the server:
`models/` contains the trained artefacts, so the app starts scoring as soon as
the container boots.

Two options below: **Blueprint** (recommended, uses `render.yaml`) and
**Manual** (if you prefer clicking through the dashboard).

---

## 1. What is already prepared

| File | Purpose |
|---|---|
| `render.yaml` | Render Blueprint: Python 3.12.5, build + start command, health check, auto-deploy on push |
| `Procfile` | Same start command, for Railway / Heroku / any process manager |
| `.python-version` | Pins Python 3.12 (the version that produced the model pickles) |
| `requirements.txt` | All runtime dependencies, pinned |
| `.streamlit/config.toml` | `headless = true`, 200 MB upload cap, file watcher off |
| `.gitignore` | Ignores secrets and the SQLite DB — **keeps** `models/*.pkl` |

The app listens on `$PORT` and `0.0.0.0`, which is what Render requires.

---

## 2. Option A — Blueprint deploy (recommended)

1. Push this project to GitHub (see §3 if you have not done that yet).
2. Go to <https://dashboard.render.com> → **New +** → **Blueprint**.
3. Connect your GitHub account and pick the repository.
4. Render reads `render.yaml` and shows one web service,
   `churn-prediction-system`, on the **Free** plan.
5. Click **Apply**. The build takes roughly 3–6 minutes (the model artefacts
   are ~107 MB).
6. When the log shows `You can now view your Streamlit app in your browser`,
   open the service URL: `https://churn-prediction-system-xxxx.onrender.com`.

No environment variables need to be typed in — `render.yaml` sets
`PYTHON_VERSION=3.12.5` for you.

## 3. Push the project to GitHub

From the project folder (the one containing `dashboard/`, `src/`, `models/`):

```cmd
git init
git add .
git commit -m "Churn prediction dashboard"
git branch -M main
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```

Confirm on github.com that these arrived:

- `render.yaml`
- `Procfile`
- `.python-version`
- `models/best_model.pkl` (~80 MB)
- `models/voting_ensemble.pkl` (~14 MB)
- `models/model_results.json`
- `data/raw/telco_churn_raw.csv`

If `models/*.pkl` is missing, an old `.gitignore` is excluding them. Restore the
one that ships with this project, or force-add:

```cmd
git add -f models/*.pkl
git commit -m "Add trained models"
git push
```

## 4. Option B — Manual deploy

1. <https://dashboard.render.com> → **New +** → **Web Service**.
2. Connect the repository.
3. Settings:

   | Field | Value |
   |---|---|
   | Language / Runtime | **Python 3** |
   | Build Command | `pip install --upgrade pip && pip install -r requirements.txt` |
   | Start Command | `streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true --server.fileWatcherType none` |
   | Health Check Path | `/_stcore/health` |
   | Instance Type | Free (or Starter, see §6) |

4. **Advanced → Add Environment Variable**:
   `PYTHON_VERSION` = `3.12.5`
5. **Create Web Service**.

---

## 5. Expected build log

```
==> Installing Python version 3.12.5
==> Running build command 'pip install --upgrade pip && pip install -r requirements.txt'
Successfully installed pandas-2.1.4 scikit-learn-1.3.2 streamlit-1.29.0 ...
==> Starting service with 'streamlit run dashboard/app.py --server.port $PORT ...'

  You can now view your Streamlit app in your browser.
  URL: http://0.0.0.0:10000
==> Your service is live 🎉
```

---

## 6. Free plan limits (read this before your demo)

| Limit | Effect on this app |
|---|---|
| **512 MB RAM** | Dashboard startup is fine (it loads one 14 MB model). The **Model Comparison & Evaluation** page trains five new models on upload and can hit the ceiling on a large CSV. |
| **Spins down after 15 min idle** | The first visit after a pause takes 30–60 s to wake. Open the URL a minute before a demo or viva. |
| **Ephemeral disk** | `data/churn.db` is recreated on every deploy/restart. Nothing user-facing depends on it. |

If you need the training page to be fast during a live demo, use the
**Starter** plan (2 GB) for that session, then downgrade.

---

## 7. Troubleshooting

**Build fails on a dependency**
The pins are a tested set (pandas 2.1.4, scikit-learn 1.3.2, xgboost 2.0.3,
streamlit 1.29.0). If `PYTHON_VERSION` is ignored, make sure the env var is set
on the service, not just in `render.yaml`.

**`FileNotFoundError: models/...` in the logs**
The pickles were not committed. See §3 (`git add -f models/*.pkl`).

**Page loads but stays on "RUNNING…"**
Render's proxy can block the Streamlit websocket. Uncomment the last two lines
of `.streamlit/config.toml` (`enableCORS`, `enableXsrfProtection`) and redeploy.

**Deploy succeeds but the service is marked unhealthy**
Check the health path is `/_stcore/health` and that the start command contains
`--server.port $PORT`.

**Uploads fail**
The upload cap is 200 MB (`maxUploadSize`). A 10,000-row CSV needs well under
1 GB of RAM, but see the memory note in §6.

**Numbers differ between pages**
They should not. Every page reads the canonical table in
`src/model_registry.py`; see §5 of `USER_MANUAL.md`.

---

## 8. Updating the deployed app

```cmd
git add -A
git commit -m "what changed"
git push
```

`autoDeploy: true` in `render.yaml` rebuilds the service on every push to
`main`. No manual redeploy needed.

---

## 9. Alternative hosts

- **Railway** — `Procfile` is already included; Railway injects `$PORT` the same
  way. Build command: `pip install -r requirements.txt`.
- **Heroku** — `Procfile` works, but the 512 MB free tier is tight; add
  `runtime.txt` with `python-3.12.5` if the build picks the wrong version.
- **Docker** — any `python:3.12-slim` image works:
  `CMD streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
