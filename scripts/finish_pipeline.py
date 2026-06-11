#!/usr/bin/env python3
"""Complete post-training steps without re-fitting models."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FIGURES_DIR, METRICS_DIR, MODELS_DIR, RANDOM_STATE, REPORTS_DIR, TEST_SIZE
from src.explainability.shap_analysis import run_shap_summary
from src.features.engineering import prepare_model_matrix
from src.strategies.retention_engine import build_action_plan, intervention_ranking


def main() -> None:
    dataset = pd.read_csv(PROJECT_ROOT / "outputs" / "modeling_dataset.csv", parse_dates=["anchor_date"])
    with open(METRICS_DIR / "retained_180_leaderboard.json", encoding="utf-8") as handle:
        retention_best = json.load(handle)[0]["model"]
    with open(METRICS_DIR / "churn_365_leaderboard.json", encoding="utf-8") as handle:
        churn_best = json.load(handle)[0]["model"]

    for target_name, best_name in [("retained_180", retention_best), ("churn_365", churn_best)]:
        x, y = prepare_model_matrix(dataset, target_name)
        _, x_test, _, y_test = train_test_split(
            x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        pipeline = joblib.load(MODELS_DIR / f"{target_name}_{best_name}.joblib")
        if best_name in {"random_forest", "xgboost", "lightgbm", "catboost"}:
            run_shap_summary(pipeline, x_test, target_name, best_name, FIGURES_DIR, METRICS_DIR)

    retention_x, _ = prepare_model_matrix(dataset, "retained_180")
    retention_model = joblib.load(MODELS_DIR / f"retained_180_{retention_best}.joblib")
    latest = dataset.sort_values("anchor_date").groupby("Donor_ID", as_index=False).tail(1)
    latest_scores = latest.set_index("Donor_ID").reset_index()
    probabilities = retention_model.predict_proba(latest_scores[retention_x.columns])[:, 1]
    latest_scores["retention_probability"] = probabilities
    build_action_plan(latest_scores).to_csv(REPORTS_DIR / "donor_action_plan.csv", index=False)
    intervention_ranking().to_csv(REPORTS_DIR / "intervention_ranking.csv", index=False)
    print("Finished post-training steps.")


if __name__ == "__main__":
    main()
