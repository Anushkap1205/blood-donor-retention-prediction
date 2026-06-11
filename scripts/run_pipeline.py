#!/usr/bin/env python3
"""End-to-end pipeline: load data, engineer features, train models, explain, segment."""

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
from src.data.cleaning import align_donations_to_donors, clean_donations, clean_donors, data_quality_report
from src.data.loader import load_donor_donation_tables
from src.eda.analysis import run_full_eda
from src.explainability.shap_analysis import run_permutation_importance, run_shap_summary
from src.features.engineering import build_observation_dataset, prepare_model_matrix
from src.models.evaluate import compare_models_statistically, plot_curves, plot_model_comparison
from src.models.train import train_all_models
from src.segmentation.rfm import build_rfm_table, segment_summary
from src.strategies.retention_engine import build_action_plan, intervention_ranking


def write_data_dictionary(report: dict, donors: pd.DataFrame, donations: pd.DataFrame) -> Path:
    dictionary = {
        "Donor_Master": {
            "primary_key": "Donor_ID",
            "foreign_keys": [],
            "rows": len(donors),
            "columns": {
                col: str(dtype) for col, dtype in donors.dtypes.items()
            },
        },
        "Donation_Register": {
            "primary_key": "Donation_ID",
            "foreign_keys": ["Donor_ID -> Donor_Master.Donor_ID"],
            "rows": len(donations),
            "columns": {
                col: str(dtype) for col, dtype in donations.dtypes.items()
            },
        },
        "quality_report": report,
    }
    path = REPORTS_DIR / "data_dictionary.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(dictionary, handle, indent=2, default=str)
    return path


def main() -> None:
    print("Loading data...")
    donors_raw, donations_raw = load_donor_donation_tables()
    donors = clean_donors(donors_raw)
    donations = clean_donations(donations_raw)
    donors, donations = align_donations_to_donors(donors, donations)
    quality = data_quality_report(donors, donations)
    write_data_dictionary(quality, donors, donations)

    print("Building observation-level dataset...")
    dataset = build_observation_dataset(donors, donations)
    dataset.to_csv(PROJECT_ROOT / "outputs" / "modeling_dataset.csv", index=False)

    print("Running EDA...")
    rfm = build_rfm_table(donors, donations)
    rfm.to_csv(REPORTS_DIR / "rfm_segments.csv", index=False)
    segment_summary(rfm).to_csv(REPORTS_DIR / "segmentation_summary.csv", index=False)
    eda_outputs = run_full_eda(donors, donations, dataset, rfm, FIGURES_DIR)

    targets = {
        "retained_180": "retained_180",
        "churn_365": "churn_365",
    }
    best_models: dict[str, str] = {}

    for target_name, target_col in targets.items():
        print(f"Training models for {target_name}...")
        x, y = prepare_model_matrix(dataset, target_col)
        leaderboard = train_all_models(x, y, target_name)
        best = leaderboard[0]
        best_models[target_name] = best.name

        comparison = compare_models_statistically(METRICS_DIR, target_name)
        comparison.to_csv(METRICS_DIR / f"{target_name}_comparison.csv", index=False)
        plot_model_comparison(comparison, target_name, FIGURES_DIR)

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        pipeline = joblib.load(best.model_path)
        plot_curves(pipeline, x_test, y_test, target_name, best.name, FIGURES_DIR)

        print(f"Explainability for best model: {best.name}")
        run_permutation_importance(
            pipeline, x_test, y_test, target_name, best.name, FIGURES_DIR, METRICS_DIR
        )
        if best.name in {"random_forest", "xgboost", "lightgbm", "catboost"}:
            run_shap_summary(pipeline, x_test, target_name, best.name, FIGURES_DIR, METRICS_DIR)

    print("Scoring donors and building retention action plan...")
    retention_x, _ = prepare_model_matrix(dataset, "retained_180")
    retention_model = joblib.load(MODELS_DIR / f"retained_180_{best_models['retained_180']}.joblib")
    latest = (
        dataset.sort_values("anchor_date")
        .groupby("Donor_ID", as_index=False)
        .tail(1)
    )
    score_cols = retention_x.columns
    latest_scores = latest.set_index("Donor_ID")
    probabilities = retention_model.predict_proba(latest_scores[score_cols])[:, 1]
    latest_scores = latest_scores.reset_index()
    latest_scores["retention_probability"] = probabilities
    action_plan = build_action_plan(latest_scores)
    action_plan.to_csv(REPORTS_DIR / "donor_action_plan.csv", index=False)
    intervention_ranking().to_csv(REPORTS_DIR / "intervention_ranking.csv", index=False)

    summary = {
        "quality_report": quality,
        "dataset_rows": len(dataset),
        "labeled_retention_rows": int(dataset["retained_180"].notna().sum()),
        "labeled_churn_rows": int(dataset["churn_365"].notna().sum()),
        "best_models": best_models,
        "eda_outputs": eda_outputs,
    }
    with open(METRICS_DIR / "pipeline_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)

    print("Pipeline complete.")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
