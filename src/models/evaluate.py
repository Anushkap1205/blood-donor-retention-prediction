"""Model comparison and error analysis utilities."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay


def compare_models_statistically(metrics_dir: Path, target_name: str) -> pd.DataFrame:
    """Create a leaderboard table from saved model metrics."""
    candidates = {"logistic_regression", "random_forest", "xgboost", "lightgbm", "catboost"}
    rows = []
    for path in sorted(metrics_dir.glob(f"{target_name}_*.json")):
        if path.name.endswith("leaderboard.json"):
            continue
        model_name = path.stem.replace(f"{target_name}_", "")
        if model_name not in candidates:
            continue
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)

        rows.append(
            {
                "model": model_name,
                "roc_auc": payload["metrics"]["roc_auc"],
                "pr_auc": payload["metrics"]["pr_auc"],
                "f1": payload["metrics"]["f1"],
                "recall": payload["metrics"]["recall"],
                "precision": payload["metrics"]["precision"],
                "accuracy": payload["metrics"]["accuracy"],
                "brier_score": payload["metrics"].get("brier_score", np.nan),
                "base_rate": payload["metrics"].get("positive_rate_test", np.nan),
                "cv_roc_auc_mean": payload["cv_metrics"]["roc_auc"]["mean"],
                "cv_roc_auc_std": payload["cv_metrics"]["roc_auc"]["std"],
            }
        )
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)



def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """Compare two classifiers with McNemar's test on paired predictions."""
    both_correct = np.sum((pred_a == y_true) & (pred_b == y_true))
    a_correct_b_wrong = np.sum((pred_a == y_true) & (pred_b != y_true))
    b_correct_a_wrong = np.sum((pred_b == y_true) & (pred_a != y_true))
    both_wrong = np.sum((pred_a != y_true) & (pred_b != y_true))
    table = np.array(
        [
            [both_correct, a_correct_b_wrong],
            [b_correct_a_wrong, both_wrong],
        ]
    )
    result = stats.mcnemar(table, exact=False, correction=True)
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "table": table.tolist(),
    }


def plot_model_comparison(leaderboard: pd.DataFrame, target_name: str, figures_dir: Path) -> Path:
    """Save bar chart comparing model ROC-AUC and recall."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(data=leaderboard, x="model", y="roc_auc", ax=axes[0], palette="Reds_r")
    axes[0].set_title(f"{target_name}: ROC-AUC")
    axes[0].tick_params(axis="x", rotation=45)
    sns.barplot(data=leaderboard, x="model", y="recall", ax=axes[1], palette="Blues_r")
    axes[1].set_title(f"{target_name}: Recall")
    axes[1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    output = figures_dir / f"{target_name}_model_comparison.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_curves(model, x_test, y_test, target_name: str, model_name: str, figures_dir: Path) -> None:
    """Save ROC, PR, and Calibration curves for a fitted pipeline."""
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    RocCurveDisplay.from_estimator(model, x_test, y_test, ax=axes[0])
    axes[0].set_title(f"{target_name} / {model_name}: ROC")
    PrecisionRecallDisplay.from_estimator(model, x_test, y_test, ax=axes[1])
    axes[1].set_title(f"{target_name} / {model_name}: PR")
    
    y_prob = model.predict_proba(x_test)[:, 1]
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy="uniform")
    brier = brier_score_loss(y_test, y_prob)
    
    axes[2].plot(prob_pred, prob_true, marker="o", linewidth=1.5, label=f"{model_name} (Brier: {brier:.4f})")
    axes[2].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    axes[2].set_xlabel("Mean predicted probability")
    axes[2].set_ylabel("Fraction of positives")
    axes[2].set_title(f"{target_name} / {model_name}: Calibration")
    axes[2].legend(loc="lower right")
    axes[2].grid(True)
    
    fig.tight_layout()
    fig.savefig(figures_dir / f"{target_name}_{model_name}_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

