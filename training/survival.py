"""Survival analysis for time-to-next-donation modeling."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import GroupShuffleSplit

from feature_store.builders import get_enhanced_feature_columns


def build_survival_frame(dataset: pd.DataFrame, censor_days: int = 365) -> pd.DataFrame:
    """Convert observation dataset to survival format with time-to-event."""
    records: list[dict] = []
    for donor_id, group in dataset.groupby("Donor_ID"):
        group = group.sort_values("anchor_date")
        for i in range(len(group) - 1):
            row = group.iloc[i]
            next_row = group.iloc[i + 1]
            delta_days = (next_row["anchor_date"] - row["anchor_date"]).days
            if delta_days <= 0:
                continue
            event = 1
            duration = min(delta_days, censor_days)
            if delta_days > censor_days:
                event = 0
            record = {
                "Donor_ID": donor_id,
                "duration_days": duration,
                "event_observed": event,
            }
            for col in get_enhanced_feature_columns():
                if col in row.index:
                    record[col] = row[col]
            records.append(record)

    if not records:
        raise ValueError("Insufficient inter-donation intervals for survival modeling.")

    frame = pd.DataFrame(records)
    numeric_cols = [c for c in get_enhanced_feature_columns() if c in frame.columns]
    for col in numeric_cols:
        if frame[col].dtype == object:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["duration_days"])
    return frame


def train_cox_model(
    survival_frame: pd.DataFrame,
    output_dir: Path,
    random_state: int = 42,
) -> dict:
    """Fit Cox proportional hazards model with donor-group holdout."""
    candidate_cols = [
        "total_donations",
        "days_since_last_donation",
        "avg_gap_days",
        "donation_velocity",
        "engagement_score",
        "loyalty_score",
        "camp_ratio",
        "Age",
    ]
    feature_cols = [c for c in candidate_cols if c in survival_frame.columns]
    model_df = survival_frame[["duration_days", "event_observed", "Donor_ID"] + feature_cols].copy()
    model_df = model_df.dropna()

    # Drop near-constant covariates that destabilize Cox PH estimation.
    for col in feature_cols:
        if model_df[col].std(ddof=0) < 1e-6:
            model_df = model_df.drop(columns=[col])
    feature_cols = [c for c in feature_cols if c in model_df.columns]
    if not feature_cols:
        raise ValueError("No usable covariates for Cox PH after variance filtering.")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_idx, test_idx = next(splitter.split(model_df, groups=model_df["Donor_ID"]))
    train = model_df.iloc[train_idx].drop(columns=["Donor_ID"])
    test = model_df.iloc[test_idx].drop(columns=["Donor_ID"])

    cph = CoxPHFitter(penalizer=0.5)
    cph.fit(train, duration_col="duration_days", event_col="event_observed")

    test_pred = cph.predict_partial_hazard(test)
    c_index = concordance_index(
        test["duration_days"],
        -test_pred,
        test["event_observed"],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    import joblib

    model_path = output_dir / "cox_ph_survival.joblib"
    joblib.dump(cph, model_path)

    km = KaplanMeierFitter()
    km.fit(train["duration_days"], train["event_observed"], label="train")
    fig, ax = plt.subplots(figsize=(8, 5))
    km.plot_survival_function(ax=ax)
    ax.set_title("Kaplan-Meier: Time to Next Donation")
    ax.set_xlabel("Days")
    fig.savefig(output_dir.parent / "figures" / "survival_km_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "concordance_index": float(c_index),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "event_rate_train": float(train["event_observed"].mean()),
        "model_path": str(model_path),
        "top_hazard_ratios": cph.summary.sort_values("exp(coef)", ascending=False)
        .head(10)[["exp(coef)", "p"]]
        .to_dict(),
    }
    metrics_path = output_dir.parent / "metrics" / "survival_cox.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)
    return summary


def predict_next_donation_date(
    cph: CoxPHFitter,
    donor_features: pd.DataFrame,
    anchor_date: pd.Timestamp,
) -> pd.Series:
    """Estimate expected next donation date from partial hazard."""
    hazard = cph.predict_partial_hazard(donor_features)
    # Lower hazard → longer expected wait; use median gap as baseline adjustment
    median_gap = donor_features.get("avg_gap_days", pd.Series([90.0])).fillna(90.0)
    expected_days = median_gap * (1.0 + hazard / hazard.median())
    return anchor_date + pd.to_timedelta(expected_days, unit="D")
