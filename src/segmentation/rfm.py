"""RFM-style donor segmentation using Recency, Frequency, and Units donated."""

from __future__ import annotations

import pandas as pd


SEGMENT_LABELS = {
    "Loyal Donors": "High frequency, recent, high volume",
    "Active Donors": "Moderate-to-high engagement with recent activity",
    "At-Risk Donors": "Previously active but recency is deteriorating",
    "Lost Donors": "Long inactive period with low recent engagement",
}


def build_rfm_table(
    donors: pd.DataFrame,
    donations: pd.DataFrame,
    reference_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Compute donor-level RFM metrics as of a reference date."""
    reference_date = reference_date or donations["Donation_Date"].max()
    donor_history = donations[donations["Donation_Date"] <= reference_date].copy()

    recency = (
        donor_history.groupby("Donor_ID")["Donation_Date"]
        .max()
        .rsub(reference_date)
        .dt.days.rename("recency_days")
    )
    frequency = donor_history.groupby("Donor_ID").size().rename("frequency")
    monetary = donor_history.groupby("Donor_ID")["Units_Collected"].sum().rename("units_donated")

    rfm = pd.concat([recency, frequency, monetary], axis=1).reset_index()
    rfm = rfm.merge(donors[["Donor_ID", "Gender", "Age", "Blood_Group", "Donor_Status"]], on="Donor_ID", how="left")

    for column in ["recency_days", "frequency", "units_donated"]:
        rfm[f"{column}_score"] = pd.qcut(rfm[column], 4, labels=[1, 2, 3, 4], duplicates="drop").astype(int)

    def assign_segment(row: pd.Series) -> str:
        if row["frequency_score"] >= 3 and row["recency_days_score"] >= 3:
            return "Loyal Donors"
        if row["recency_days_score"] >= 3:
            return "Active Donors"
        if row["recency_days_score"] == 2:
            return "At-Risk Donors"
        return "Lost Donors"

    rfm["segment"] = rfm.apply(assign_segment, axis=1)
    return rfm


def segment_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    """Aggregate business metrics by donor segment."""
    summary = (
        rfm.groupby("segment")
        .agg(
            donors=("Donor_ID", "count"),
            avg_recency_days=("recency_days", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_units_donated=("units_donated", "mean"),
        )
        .reset_index()
    )
    summary["interpretation"] = summary["segment"].map(SEGMENT_LABELS)
    return summary.sort_values("donors", ascending=False)
