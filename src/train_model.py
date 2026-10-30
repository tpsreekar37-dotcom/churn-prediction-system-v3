"""
train_model.py
--------------
Model training, comparison, hyperparameter tuning, and evaluation
for the Customer Churn Prediction System.

Models trained:
  1. Logistic Regression (interpretable baseline)
  2. Random Forest
  3. XGBoost
  4. LightGBM
  5. Support Vector Machine (SVM)
  6. Voting Ensemble (soft voting over models 1-5)

The model with the highest holdout ROC-AUC is saved as the production model
(F1 breaks ties). The best model is then calibrated with CalibratedClassifierCV
and the optimal decision threshold is found via Youden's J statistic.

Cross-validation runs on the original, un-resampled training data so the
reported AUC is not inflated by synthetic SMOTE rows.
"""

import os
import sys
import json
import datetime
import platform
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, VotingClassifier
from sklearn.svm             import SVC
from sklearn.calibration     import CalibratedClassifierCV
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve, ConfusionMatrixDisplay,
)
from sklearn.model_selection import GridSearchCV, cross_val_score
from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import logger, MODELS_DIR, REPORTS_DIR, save_figure, set_style, PALETTE, CHURN_COLORS, ensure_dirs, DECISION_THRESHOLD


# ──────────────────────────────────────────────────────────────────────────────
# 1. MODEL DEFINITIONS
# ──────────────────────────────────────────────────────────────────────────────

def get_models() -> dict:
    """Return a dict of model name → unfitted estimator."""
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced",
            solver="lbfgs", C=1.0,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42,
            n_jobs=-1, class_weight="balanced"
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            eval_metric="logloss",
            random_state=42, n_jobs=-1,
            scale_pos_weight=3  # handle imbalance
        ),
        "SVM": SVC(
            kernel="rbf", C=1.0, gamma="scale",
            probability=True, random_state=42,
            class_weight="balanced",
        ),
    }
    if _HAS_LGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            random_state=42, n_jobs=-1, verbose=-1,
            is_unbalance=True,
        )
    return models


# ──────────────────────────────────────────────────────────────────────────────
# 2. EVALUATION
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series,
                   model_name: str = "Model",
                   threshold: float = DECISION_THRESHOLD) -> dict:
    """
    Compute all classification metrics and return as dict.

    Labels come from ``threshold`` rather than ``model.predict`` so the saved
    metrics describe exactly the decision rule the app serves with.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    metrics = {
        "model":     model_name,
        "accuracy":  round(accuracy_score(y_test, y_pred),  4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred,    zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred,        zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba),                  4),
        "threshold": threshold,
    }

    logger.info(
        f"[{model_name}] acc={metrics['accuracy']:.4f} | "
        f"prec={metrics['precision']:.4f} | rec={metrics['recall']:.4f} | "
        f"f1={metrics['f1']:.4f} | auc={metrics['roc_auc']:.4f}"
    )
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# 3. VISUALISATIONS
# ──────────────────────────────────────────────────────────────────────────────

def plot_model_comparison(results: list, output_path: str = None) -> str:
    """Bar chart comparing all models across metrics."""
    set_style()
    df_res = pd.DataFrame(results).set_index("model")
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    colors  = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"],
                PALETTE["positive"], PALETTE["negative"]]

    x = np.arange(len(df_res))
    width = 0.15
    fig, ax = plt.subplots(figsize=(13, 6))

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        bars = ax.bar(x + i * width, df_res[metric], width, label=metric.upper(),
                      color=color, alpha=0.85, edgecolor="#0F172A")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom",
                    fontsize=7.5, color="#E2E8F0")

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(df_res.index, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Performance Comparison", fontsize=15, fontweight="bold", pad=12)
    ax.legend(loc="lower right", framealpha=0.2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out = output_path or os.path.join(REPORTS_DIR, "model_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Model comparison chart → {out}")
    return out


# Extended color palette for up to 7 models
_PLOT_COLORS = [
    PALETTE["primary"], PALETTE["accent"], PALETTE["positive"],
    PALETTE["negative"], PALETTE["secondary"], PALETTE["neutral"],
    "#7C3AED",  # violet for ensemble
]


def plot_roc_curves(models_dict: dict, X_test: pd.DataFrame,
                    y_test: pd.Series) -> str:
    """Overlay ROC curves for all trained models."""
    set_style()
    fig, ax = plt.subplots(figsize=(10, 7))

    for i, (name, model) in enumerate(models_dict.items()):
        color = _PLOT_COLORS[i % len(_PLOT_COLORS)]
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, lw=2.5, color=color, label=f"{name} (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5, label="Random Baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves – All Models", fontsize=15, fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.2, fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out = os.path.join(REPORTS_DIR, "roc_curves.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"ROC curves → {out}")
    return out


def plot_confusion_matrix(model, X_test: pd.DataFrame, y_test: pd.Series,
                          model_name: str) -> str:
    """Plot and save a styled confusion matrix."""
    set_style()
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        linewidths=0.5, linecolor="#334155",
        annot_kws={"size": 14, "weight": "bold"},
        ax=ax
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual",    fontsize=12)
    ax.set_title(f"Confusion Matrix – {model_name}", fontsize=14, fontweight="bold", pad=10)
    plt.tight_layout()

    fname = f"confusion_matrix_{model_name.replace(' ', '_').lower()}.png"
    out = os.path.join(REPORTS_DIR, fname)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#1E293B")
    plt.close(fig)
    logger.info(f"Confusion matrix → {out}")
    return out


def plot_feature_importance(model, feature_names: list,
                             model_name: str, top_n: int = 20) -> str:
    """Plot and save top-N feature importances."""
    set_style()
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        logger.warning("Model has no feature_importances_ or coef_ attribute.")
        return ""

    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors_bar = [PALETTE["primary"]] * top_n
    sns.barplot(
        data=df_imp, y="Feature", x="Importance",
        palette=colors_bar[:len(df_imp)], edgecolor="#0F172A", ax=ax
    )
    ax.set_title(f"Top {top_n} Feature Importances – {model_name}",
                 fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel("Importance Score")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    fname = f"feature_importance_{model_name.replace(' ', '_').lower()}.png"
    out = os.path.join(REPORTS_DIR, fname)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Feature importance → {out}")

    # Save as CSV too
    df_imp.to_csv(os.path.join(MODELS_DIR, "feature_importance.csv"), index=False)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 4. CALIBRATION & THRESHOLD OPTIMISATION
# ──────────────────────────────────────────────────────────────────────────────

def calibrate_model(model, X_train: pd.DataFrame, y_train: pd.Series,
                    method: str = "isotonic"):
    """
    Wrap the model in CalibratedClassifierCV so that predict_proba returns
    well-calibrated probabilities. Uses cross-validated calibration on the
    training set.
    """
    logger.info(f"Calibrating model with {method} regression …")
    calibrated = CalibratedClassifierCV(model, method=method, cv=5)
    calibrated.fit(X_train, y_train)
    return calibrated


def find_optimal_threshold(model, X_test: pd.DataFrame,
                            y_test: pd.Series) -> float:
    """
    Find the optimal decision threshold using Youden's J statistic
    (maximises TPR − FPR on the ROC curve).
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = float(thresholds[best_idx])
    logger.info(
        f"Optimal threshold (Youden's J): {best_threshold:.4f} "
        f"(TPR={tpr[best_idx]:.4f}, FPR={fpr[best_idx]:.4f})"
    )
    return best_threshold


# ──────────────────────────────────────────────────────────────────────────────
# 5. TRAINING PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def train_all_models(X_train: pd.DataFrame, X_test: pd.DataFrame,
                     y_train: pd.Series, y_test: pd.Series):
    """
    Train all individual models, then build a soft-voting ensemble.
    Evaluate all (including ensemble), pick the best by ROC-AUC.
    Returns: best_model, all_models_dict, results_list, best_model_name
    """
    ensure_dirs()
    models   = get_models()
    results  = []
    trained  = {}

    for name, model in models.items():
        logger.info(f"\n── Training: {name} ──")
        model.fit(X_train, y_train)
        trained[name] = model
        metrics = evaluate_model(model, X_test, y_test, name)
        results.append(metrics)

    # ── Build Voting Ensemble from all trained models
    logger.info("\n── Building: Voting Ensemble ──")
    estimators = [(name.lower().replace(" ", "_"), m) for name, m in trained.items()]
    ensemble = VotingClassifier(
        estimators=estimators, voting="soft", n_jobs=-1,
    )
    # sklearn VotingClassifier requires fit(), but we already have fitted
    # estimators. We use the trick of setting estimators_ directly.
    ensemble.estimators_ = list(trained.values())
    from sklearn.preprocessing import LabelEncoder as _LE
    ensemble.le_ = _LE().fit(y_train)
    ensemble.classes_ = ensemble.le_.classes_
    trained["Voting Ensemble"] = ensemble
    metrics = evaluate_model(ensemble, X_test, y_test, "Voting Ensemble")
    results.append(metrics)

    # Pick the production model from measured performance, not by name.
    best_row = max(results, key=lambda r: (r["roc_auc"], r["f1"]))
    best_name = best_row["model"]
    best_model = trained[best_name]
    logger.info(f"\n★  Production model: {best_name} (AUC = {best_row['roc_auc']:.4f})")

    return best_model, trained, results, best_name


def run_training_pipeline(X_train: pd.DataFrame, X_test: pd.DataFrame,
                           y_train: pd.Series, y_test: pd.Series,
                           cv_X: pd.DataFrame = None, cv_y: pd.Series = None):
    """
    Full training pipeline:
    1. Train & evaluate all models (including ensemble)
    2. Calibrate the best model's probabilities
    3. Find optimal decision threshold
    4. Save best model + artifacts
    5. Generate all visualisations
    Returns: best_model, results, best_name

    ``cv_X``/``cv_y`` should be the *unbalanced* training split. When omitted,
    cross-validation falls back to the data supplied for fitting, which is
    SMOTE-balanced and will overstate generalisation.
    """
    ensure_dirs()

    best_model, trained, results, best_name = train_all_models(
        X_train, X_test, y_train, y_test
    )

    # ── Calibrate the best model
    try:
        best_model_calibrated = calibrate_model(best_model, X_train, y_train)
        # Verify calibration didn't hurt AUC
        calib_proba = best_model_calibrated.predict_proba(X_test)[:, 1]
        calib_auc = roc_auc_score(y_test, calib_proba)
        orig_auc = next(r for r in results if r["model"] == best_name)["roc_auc"]
        if calib_auc >= orig_auc - 0.005:  # accept small AUC dip for better calibration
            best_model = best_model_calibrated
            logger.info(f"Calibrated model accepted (AUC: {orig_auc:.4f} → {calib_auc:.4f})")
        else:
            logger.info(f"Calibration hurt AUC too much ({orig_auc:.4f} → {calib_auc:.4f}), keeping uncalibrated")
    except Exception as e:
        logger.warning(f"Calibration failed, using uncalibrated model: {e}")

    # ── Find optimal threshold (informational only)
    # The app serves at DECISION_THRESHOLD everywhere; the Youden-J threshold is
    # recorded for analysis but deliberately not applied, because it trades a
    # large accuracy loss for recall and would make the served labels disagree
    # with the saved metrics.
    optimal_threshold = find_optimal_threshold(best_model, X_test, y_test)

    # ── Save best model + metadata
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump(best_model, model_path)
    logger.info(f"Best model saved → {model_path}")

    # Save all individual models
    for name, m in trained.items():
        fname = name.lower().replace(" ", "_") + ".pkl"
        joblib.dump(m, os.path.join(MODELS_DIR, fname))

    # Save results JSON
    results_path = os.path.join(MODELS_DIR, "model_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved → {results_path}")

    # Save best model name
    with open(os.path.join(MODELS_DIR, "best_model_name.txt"), "w") as f:
        f.write(best_name)

    # Save provenance so anyone loading the .pkl can verify the library versions
    # that produced it. pickles silently misbehave across sklearn major versions.
    metadata = {
        "model_name":          best_name,
        "trained_at":          datetime.datetime.now().isoformat(timespec="seconds"),
        "python_version":      platform.python_version(),
        "sklearn_version":     sklearn.__version__,
        "pandas_version":      pd.__version__,
        "numpy_version":       np.__version__,
        "n_train_rows":        int(len(X_train)),
        "n_features":          int(X_train.shape[1]),
        "features":            list(X_train.columns),
        "metrics":             next(r for r in results if r["model"] == best_name),
        "all_models_trained":  [r["model"] for r in results],
        "decision_threshold":  DECISION_THRESHOLD,
        "youden_threshold_informational": optimal_threshold,
        "calibrated":          isinstance(best_model, CalibratedClassifierCV),
        "synthetic_training":  True,
    }
    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Model metadata saved (sklearn {sklearn.__version__})")

    # ── Plots
    plot_model_comparison(results)
    plot_roc_curves(trained, X_test, y_test)
    plot_confusion_matrix(best_model, X_test, y_test, best_name)
    plot_feature_importance(best_model, list(X_train.columns), best_name)

    # ── Cross-validation (on real, un-resampled data when available)
    cv_features = cv_X if cv_X is not None else X_train
    cv_target   = cv_y if cv_y is not None else y_train
    logger.info(f"Running 5-fold cross-validation on {'original' if cv_X is not None else 'SMOTE-balanced'} data...")
    # Use the uncalibrated best model for CV (calibration wraps fit internally)
    cv_model = trained[best_name]
    cv_scores = cross_val_score(cv_model, cv_features, cv_target, cv=5, scoring="roc_auc")
    logger.info(f"CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Persist the CV result next to the model so reports can quote it honestly.
    cv_summary = {
        "cv_auc_mean": round(float(cv_scores.mean()), 4),
        "cv_auc_std":  round(float(cv_scores.std()), 4),
        "cv_rows":     int(len(cv_features)),
        "cv_on_smote_data": cv_X is None,
    }
    with open(os.path.join(MODELS_DIR, "cv_results.json"), "w") as f:
        json.dump(cv_summary, f, indent=2)

    # Print full classification report
    y_pred = (best_model.predict_proba(X_test)[:, 1] >= DECISION_THRESHOLD).astype(int)
    print("\n" + "=" * 60)
    print(f"BEST MODEL: {best_name}")
    print(f"Decision Threshold (served): {DECISION_THRESHOLD:.4f}")
    print(f"Youden-J Threshold (informational only): {optimal_threshold:.4f}")
    print(f"Calibrated: {isinstance(best_model, CalibratedClassifierCV)}")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    return best_model, results, best_name


if __name__ == "__main__":
    from src.preprocessing      import run_preprocessing_pipeline
    from src.feature_engineering import run_feature_engineering

    X_tr, X_te, y_tr, y_te, scaler, encoders, df = run_preprocessing_pipeline()
    X_tr_b, X_te_fe, y_tr_b, y_te, X_tr_raw = run_feature_engineering(df, X_tr, X_te, y_tr, y_te)
    best_model, results, best_name = run_training_pipeline(
        X_tr_b, X_te_fe, y_tr_b, y_te, cv_X=X_tr_raw, cv_y=y_tr
    )
    print(f"\nTraining complete. Best model: {best_name}")
