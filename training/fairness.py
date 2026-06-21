"""Fairness and bias auditing for donor retention models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score


def _group_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> dict:
    pos_rate = float(y_true.mean()) if len(y_true) else 0.0
    pred_pos_rate = float(y_pred.mean()) if len(y_pred) else 0.0
    tpr = float((y_pred[y_true == 1] == 1).mean()) if (y_true == 1).any() else 0.0
    try:
        auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan
    except ValueError:
        auc = np.nan
    try:
        prob_true, _ = calibration_curve(y_true, y_prob, n_bins=5, strategy="quantile")
        calibration_error = float(np.mean(np.abs(prob_true - np.linspace(0, 1, len(prob_true)))))
    except ValueError:
        calibration_error = np.nan
    return {
        "n": int(len(y_true)),
        "positive_rate": pos_rate,
        "predicted_positive_rate": pred_pos_rate,
        "tpr_at_default_threshold": tpr,
        "roc_auc": auc,
        "calibration_error": calibration_error,
    }


def audit_fairness(
    y_true: pd.Series,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    groups: pd.DataFrame,
    protected_columns: list[str],
) -> dict:
    """Compute demographic parity, equal opportunity, and calibration by group."""
    report: dict = {"overall": _group_metrics(y_true.values, y_prob, y_pred), "by_group": {}}

    for col in protected_columns:
        if col not in groups.columns:
            continue
        col_report = {}
        aligned_groups = groups.loc[y_true.index, col]
        for group_val in aligned_groups.unique():
            mask = (aligned_groups == group_val).values
            col_report[str(group_val)] = _group_metrics(
                y_true.values[mask],
                y_prob[mask],
                y_pred[mask],
            )
        report["by_group"][col] = col_report

        # Demographic parity difference
        pred_rates = [v["predicted_positive_rate"] for v in col_report.values()]
        report[f"{col}_demographic_parity_gap"] = float(max(pred_rates) - min(pred_rates)) if pred_rates else 0.0

        # Equal opportunity gap (TPR difference)
        tprs = [v["tpr_at_default_threshold"] for v in col_report.values() if v["n"] > 0]
        report[f"{col}_equal_opportunity_gap"] = float(max(tprs) - min(tprs)) if len(tprs) > 1 else 0.0

    return report


def recommend_mitigation(fairness_report: dict, threshold: float = 0.10) -> list[str]:
    """Suggest mitigation strategies when bias gaps exceed threshold."""
    recommendations = []
    for key, value in fairness_report.items():
        if key.endswith("_demographic_parity_gap") and isinstance(value, float) and value > threshold:
            attr = key.replace("_demographic_parity_gap", "")
            recommendations.append(
                f"Demographic parity gap for {attr} ({value:.3f}) exceeds {threshold}. "
                f"Consider threshold adjustment per group or reweighing training samples."
            )
        if key.endswith("_equal_opportunity_gap") and isinstance(value, float) and value > threshold:
            attr = key.replace("_equal_opportunity_gap", "")
            recommendations.append(
                f"Equal opportunity gap for {attr} ({value:.3f}) exceeds {threshold}. "
                f"Consider equalized odds post-processing or group-specific calibration."
            )
    if not recommendations:
        recommendations.append("No significant fairness gaps detected at current threshold.")
    return recommendations


def save_fairness_report(report: dict, output_path: Path, mitigation: list[str]) -> None:
    payload = {"audit": report, "mitigation_recommendations": mitigation}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
