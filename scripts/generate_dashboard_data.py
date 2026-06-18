#!/usr/bin/env python3
"""Compile prediction results and model leaderboards into dashboard/data.js."""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"


def main():
    print("Compiling dashboard data...")
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Scored Donors Action Plan
    action_plan_path = REPORTS_DIR / "donor_action_plan.csv"
    if not action_plan_path.exists():
        print(f"Error: {action_plan_path} not found.")
        return

    df = pd.read_csv(action_plan_path)

    # Extract relevant dashboard columns to keep size compact
    cols_to_keep = [
        "Donor_ID",
        "Gender",
        "Age",
        "retention_probability",
        "churn_probability",
        "risk_category",
        "recommended_interventions",
        "Group",
        "Assignment_Date",
    ]
    # Handle missing columns if any
    available_cols = [c for c in cols_to_keep if c in df.columns]
    donors_clean = df[available_cols].copy()

    # Convert probability columns to rounded floats, and strings for JSON compatibility
    if "retention_probability" in donors_clean.columns:
        donors_clean["retention_probability"] = donors_clean["retention_probability"].round(4)
    if "churn_probability" in donors_clean.columns:
        donors_clean["churn_probability"] = donors_clean["churn_probability"].round(4)

    # Convert recommended_interventions from string representation of list to clean string
    if "recommended_interventions" in donors_clean.columns:
        def clean_interventions(val):
            if isinstance(val, str) and val.startswith("["):
                try:
                    parsed = json.loads(val.replace("'", '"'))
                    return ", ".join(parsed)
                except Exception:
                    return val.strip("[]'\"")
            return str(val)
        donors_clean["recommended_interventions"] = donors_clean["recommended_interventions"].apply(clean_interventions)

    # Convert to dictionary list
    donors_list = donors_clean.to_dict(orient="records")

    # 2. Compute Summary Metrics
    total_donors = len(df)
    avg_retention = df["retention_probability"].mean() if "retention_probability" in df.columns else 0.0
    avg_churn = df["churn_probability"].mean() if "churn_probability" in df.columns else 0.0

    # Risk category counts
    risk_counts = {}
    if "risk_category" in df.columns:
        risk_counts = df["risk_category"].value_counts().to_dict()

    # Intervention counts
    intervention_counts = {}
    if "recommended_interventions" in donors_clean.columns:
        intervention_counts = donors_clean["recommended_interventions"].value_counts().to_dict()

    # Group counts
    group_counts = {}
    if "Group" in df.columns:
        group_counts = df["Group"].value_counts().to_dict()

    summary_stats = {
        "total_donors": total_donors,
        "avg_retention_probability": round(float(avg_retention), 4),
        "avg_churn_probability": round(float(avg_churn), 4),
        "risk_counts": risk_counts,
        "intervention_counts": {k: int(v) for k, v in intervention_counts.items()},
        "group_counts": {k: int(v) for k, v in group_counts.items()},
    }

    # 3. Load Model Comparison Dataframes
    leaderboard_180 = []
    comparison_180_path = METRICS_DIR / "retained_180_comparison.csv"
    if comparison_180_path.exists():
        comp_df = pd.read_csv(comparison_180_path)
        leaderboard_180 = comp_df.to_dict(orient="records")

    leaderboard_365 = []
    comparison_365_path = METRICS_DIR / "churn_365_comparison.csv"
    if comparison_365_path.exists():
        comp_df = pd.read_csv(comparison_365_path)
        leaderboard_365 = comp_df.to_dict(orient="records")

    # 4. Save to JavaScript Data File
    payload = {
        "summary": summary_stats,
        "leaderboard_180": leaderboard_180,
        "leaderboard_365": leaderboard_365,
        "donors": donors_list,
    }

    output_path = DASHBOARD_DIR / "data.js"
    with open(output_path, "w", encoding="utf-8") as out:
        out.write(f"window.dashboardData = {json.dumps(payload, indent=2)};\n")

    print(f"Successfully compiled dashboard data to {output_path}")


if __name__ == "__main__":
    main()
