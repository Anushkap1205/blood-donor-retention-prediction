"""Research-guided feature engineering for donor retention prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import AGE_BINS, AGE_LABELS, CHURN_WINDOW_DAYS, RETENTION_WINDOW_DAYS


def _gap_stats(dates: pd.Series) -> dict[str, float]:
    if len(dates) < 2:
        return {
            "avg_gap_days": np.nan,
            "std_gap_days": np.nan,
            "min_gap_days": np.nan,
            "max_gap_days": np.nan,
        }
    gaps = dates.sort_values().diff().dt.days.dropna()
    return {
        "avg_gap_days": float(gaps.mean()),
        "std_gap_days": float(gaps.std(ddof=0)) if len(gaps) > 0 else np.nan,
        "min_gap_days": float(gaps.min()),
        "max_gap_days": float(gaps.max()),
    }


def _count_in_window(dates: pd.Series, anchor: pd.Timestamp, days: int) -> int:
    start = anchor - pd.Timedelta(days=days)
    return int(((dates > start) & (dates <= anchor)).sum())


def build_observation_dataset(
    donors: pd.DataFrame,
    donations: pd.DataFrame,
    retention_days: int = RETENTION_WINDOW_DAYS,
    churn_days: int = CHURN_WINDOW_DAYS,
) -> pd.DataFrame:
    """
    Create one row per donation event with point-in-time features and labels.

    Labels are only computed when the full prediction window is observable
    within the dataset horizon (right-censoring excluded).
    """
    donors = donors.set_index("Donor_ID")
    max_date = donations["Donation_Date"].max()
    records: list[dict] = []

    for donor_id, donor_donations in donations.groupby("Donor_ID"):
        donor_donations = donor_donations.sort_values("Donation_Date").reset_index(drop=True)
        donor = donors.loc[donor_id]
        registration_date = pd.to_datetime(donor["Registration_Date"])
        future_dates = donor_donations["Donation_Date"].tolist()

        for idx, row in donor_donations.iterrows():
            anchor = row["Donation_Date"]
            history = donor_donations.iloc[: idx + 1]
            history_dates = history["Donation_Date"]
            gaps = _gap_stats(history_dates)

            retained_end = anchor + pd.Timedelta(days=retention_days)
            churn_end = anchor + pd.Timedelta(days=churn_days)
            future_after_anchor = [d for d in future_dates if d > anchor]

            retained_180 = np.nan
            churn = np.nan
            if retained_end <= max_date:
                retained_180 = int(
                    any(anchor < d <= retained_end for d in future_after_anchor)
                )
            if churn_end <= max_date:
                churn = int(
                    not any(anchor < d <= churn_end for d in future_after_anchor)
                )

            camp_count = int((history["Collection_Source"] == "Camp").sum())
            walkin_count = int((history["Collection_Source"] == "Walk-in").sum())
            total_donations = len(history)
            tenure_days = (anchor - registration_date).days
            days_since_last = 0 if total_donations == 1 else int(
                (anchor - history_dates.iloc[-2]).days
            )

            records.append(
                {
                    "Donor_ID": donor_id,
                    "Donation_ID": row["Donation_ID"],
                    "anchor_date": anchor,
                    "Gender": donor["Gender"],
                    "Age": donor["Age"],
                    "Blood_Group": donor["Blood_Group"],
                    "Donation_Frequency_Label": donor["Donation_Frequency"],
                    "Donor_Status_Label": donor["Donor_Status"],
                    "days_since_registration": max(tenure_days, 0),
                    "days_since_last_donation": days_since_last,
                    "total_donations": total_donations,
                    "donations_last_3_months": _count_in_window(history_dates, anchor, 90),
                    "donations_last_6_months": _count_in_window(history_dates, anchor, 180),
                    "donations_last_12_months": _count_in_window(history_dates, anchor, 365),
                    "avg_gap_days": gaps["avg_gap_days"],
                    "std_gap_days": gaps["std_gap_days"],
                    "min_gap_days": gaps["min_gap_days"],
                    "max_gap_days": gaps["max_gap_days"],
                    "tenure_days": tenure_days,
                    "years_as_donor": tenure_days / 365.25,
                    "camp_donation_count": camp_count,
                    "walkin_donation_count": walkin_count,
                    "camp_ratio": camp_count / total_donations,
                    "walkin_ratio": walkin_count / total_donations,
                    "total_units_donated": float(history["Units_Collected"].sum()),
                    "average_units_per_donation": float(history["Units_Collected"].mean()),
                    "donation_velocity": total_donations / max(tenure_days / 365.25, 1 / 365.25),
                    "recent_activity_score": _count_in_window(history_dates, anchor, 180)
                    / max(total_donations, 1),
                    "is_first_donation": int(total_donations == 1),
                    "donation_type_whole_blood_share": float(
                        (history["Donation_Type"] == "Whole Blood").mean()
                    ),
                    "retained_180": retained_180,
                    "churn_365": churn,
                }
            )

    dataset = pd.DataFrame(records)
    dataset["age_group"] = pd.cut(
        dataset["Age"],
        bins=AGE_BINS,
        labels=AGE_LABELS,
        right=True,
        include_lowest=True,
    )
    return dataset


def get_feature_columns() -> list[str]:
    """Numeric and categorical model features derived from transaction logs."""
    return [
        "days_since_registration",
        "days_since_last_donation",
        "total_donations",
        "donations_last_3_months",
        "donations_last_6_months",
        "donations_last_12_months",
        "avg_gap_days",
        "std_gap_days",
        "min_gap_days",
        "max_gap_days",
        "tenure_days",
        "years_as_donor",
        "camp_donation_count",
        "walkin_donation_count",
        "camp_ratio",
        "walkin_ratio",
        "total_units_donated",
        "average_units_per_donation",
        "donation_velocity",
        "recent_activity_score",
        "is_first_donation",
        "donation_type_whole_blood_share",
        "Age",
        "Gender",
        "Blood_Group",
        "age_group",
        "Donation_Frequency_Label",
    ]


def prepare_model_matrix(
    dataset: pd.DataFrame, target: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Filter labeled rows and return feature matrix with target."""
    labeled = dataset.dropna(subset=[target]).copy()
    features = get_feature_columns()
    x = labeled[features].copy()
    y = labeled[target].astype(int)
    return x, y
