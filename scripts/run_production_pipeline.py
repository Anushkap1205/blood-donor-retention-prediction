"""Production orchestration pipeline — donor retention intelligence system v2."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from feature_store.builders import build_feature_store_dataset, prepare_enhanced_matrix
from inference.explainer import batch_explain
from inference.scorer import DonorScorer
from recommendation_engine.engine import build_full_action_plan
from src.config import FIGURES_DIR, METRICS_DIR, MODELS_DIR, REPORTS_DIR
from src.data.cleaning import align_donations_to_donors, clean_donations, clean_donors
from src.data.loader import load_donor_donation_tables
from src.explainability.shap_analysis import run_permutation_importance, run_shap_summary
from src.models.evaluate import compare_models_statistically, plot_curves, plot_model_comparison
from src.models.train import train_all_models
from training.fairness import audit_fairness, recommend_mitigation, save_fairness_report
from training.survival import build_survival_frame, train_cox_model
from training.uplift import run_uplift_analysis


def load_config() -> dict:
    config_path = PROJECT_ROOT / "config" / "pipeline.yaml"
    with open(config_path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    config = load_config()
    print("=== Donor Retention Intelligence Pipeline v2 ===")

    donors_raw, donations_raw = load_donor_donation_tables()
    donors = clean_donors(donors_raw)
    donations = clean_donations(donations_raw)
    donors, donations = align_donations_to_donors(donors, donations)

    print("Building enhanced feature store...")
    dataset = build_feature_store_dataset(donors, donations)
    artifact_path = PROJECT_ROOT / "feature_store" / "artifacts" / "observation_dataset.parquet"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(artifact_path, index=False)

    targets = {
        "retained_180": config["targets"]["retention"]["name"],
        "churn_365": config["targets"]["churn"]["name"],
    }
    best_models: dict[str, str] = {}

    for target_name, target_col in targets.items():
        print(f"Training {target_name}...")
        x, y = prepare_enhanced_matrix(dataset, target_col)
        groups = dataset.loc[x.index, "Donor_ID"]
        leaderboard = train_all_models(x, y, target_name, groups=groups)
        best_models[target_name] = leaderboard[0].name

        comparison = compare_models_statistically(METRICS_DIR, target_name)
        comparison.to_csv(METRICS_DIR / f"{target_name}_comparison.csv", index=False)
        plot_model_comparison(comparison, target_name, FIGURES_DIR)

        from sklearn.model_selection import StratifiedGroupKFold
        from src.config import RANDOM_STATE

        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        train_idx, test_idx = next(sgkf.split(x, y, groups))
        x_test = x.iloc[test_idx]
        y_test = y.iloc[test_idx]
        pipeline = joblib.load(leaderboard[0].model_path)
        plot_curves(pipeline, x_test, y_test, target_name, leaderboard[0].name, FIGURES_DIR)
        run_permutation_importance(pipeline, x_test, y_test, target_name, leaderboard[0].name, FIGURES_DIR, METRICS_DIR)
        if leaderboard[0].name in {"random_forest", "xgboost", "lightgbm", "catboost"}:
            run_shap_summary(pipeline, x_test, target_name, leaderboard[0].name, FIGURES_DIR, METRICS_DIR)

        # Fairness audit on retention/churn champion
        y_prob = pipeline.predict_proba(x_test)[:, 1]
        y_pred = pipeline.predict(x_test)
        group_df = dataset.loc[x_test.index, ["Gender", "age_group"]]
        fairness = audit_fairness(y_test, y_prob, y_pred, group_df, ["Gender", "age_group"])
        mitigation = recommend_mitigation(fairness, threshold=config["fairness"]["mitigation_threshold"])
        save_fairness_report(fairness, METRICS_DIR / f"{target_name}_fairness.json", mitigation)

    print("Survival analysis...")
    try:
        survival_frame = build_survival_frame(dataset)
        train_cox_model(survival_frame, MODELS_DIR)
    except Exception as exc:
        print(f"Survival analysis skipped: {exc}")

    print("Scoring donors...")
    scorer = DonorScorer(MODELS_DIR)
    scorer.load(best_models["retained_180"], best_models["churn_365"])
    scored = scorer.score_latest(dataset)

    print("Building recommendations...")
    action_plan = build_full_action_plan(scored)
    action_plan.to_csv(REPORTS_DIR / "donor_action_plan.csv", index=False)

    print("Uplift / causal analysis...")
    x_ret, y_ret = prepare_enhanced_matrix(dataset, "retained_180")
    groups = dataset.loc[x_ret.index, "Donor_ID"]
    run_uplift_analysis(scored, x_ret, y_ret, groups, REPORTS_DIR)

    print("Local SHAP explanations...")
    from feature_store.builders import get_enhanced_feature_columns

    batch_explain(
        MODELS_DIR / f"retained_180_{best_models['retained_180']}.joblib",
        scored,
        get_enhanced_feature_columns(),
        REPORTS_DIR / "donor_explanations.json",
    )

    summary = {
        "pipeline_version": config["project"]["version"],
        "dataset_rows": len(dataset),
        "best_models": best_models,
        "feature_count": len(get_enhanced_feature_columns()),
    }
    with open(METRICS_DIR / "pipeline_summary_v2.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("Pipeline v2 complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
