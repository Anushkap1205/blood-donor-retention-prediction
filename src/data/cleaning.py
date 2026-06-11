"""Data cleaning and quality checks for donor retention modeling."""

from __future__ import annotations

import pandas as pd


DATE_COLUMNS = {
    "donors": ["Registration_Date", "Last_Donation_Date"],
    "donations": ["Donation_Date"],
}


def _parse_dates(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], errors="coerce")
    return out


def _standardize_text(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)


def clean_donors(donors: pd.DataFrame) -> pd.DataFrame:
    """Clean donor master records."""
    df = _parse_dates(donors, DATE_COLUMNS["donors"])
    df = df.drop_duplicates(subset=["Donor_ID"]).copy()
    for column in ["Gender", "Blood_Group", "Donation_Frequency", "Donor_Status"]:
        if column in df.columns:
            df[column] = _standardize_text(df[column])
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    return df.reset_index(drop=True)


def clean_donations(donations: pd.DataFrame) -> pd.DataFrame:
    """Clean donation transaction records."""
    df = _parse_dates(donations, DATE_COLUMNS["donations"])
    df = df.drop_duplicates(subset=["Donation_ID"]).copy()
    for column in ["Blood_Group", "Donation_Type", "Collection_Source"]:
        if column in df.columns:
            df[column] = _standardize_text(df[column])
    df["Units_Collected"] = pd.to_numeric(df["Units_Collected"], errors="coerce")
    df = df.sort_values(["Donor_ID", "Donation_Date", "Donation_ID"]).reset_index(drop=True)
    return df


def align_donations_to_donors(
    donors: pd.DataFrame, donations: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only donations with a matching donor record."""
    valid_ids = set(donors["Donor_ID"])
    aligned = donations[donations["Donor_ID"].isin(valid_ids)].copy()
    return donors, aligned


def data_quality_report(donors: pd.DataFrame, donations: pd.DataFrame) -> dict:
    """Summarize data quality metrics used in EDA and reports."""
    orphan_ids = set(donations["Donor_ID"]) - set(donors["Donor_ID"])
    donors_without_donations = set(donors["Donor_ID"]) - set(donations["Donor_ID"])
    return {
        "donor_rows": len(donors),
        "donation_rows": len(donations),
        "duplicate_donations": int(donations.duplicated(subset=["Donation_ID"]).sum()),
        "orphan_donation_donor_ids": len(orphan_ids),
        "donors_without_donations": len(donors_without_donations),
        "donor_null_counts": donors.isnull().sum().to_dict(),
        "donation_null_counts": donations.isnull().sum().to_dict(),
        "donation_date_range": (
            donations["Donation_Date"].min(),
            donations["Donation_Date"].max(),
        ),
    }
