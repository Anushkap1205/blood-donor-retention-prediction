"""Exploratory data analysis and publication-quality visualizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", context="talk")


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_missing_values(donors: pd.DataFrame, donations: pd.DataFrame, figures_dir: Path) -> Path:
    missing = pd.concat(
        [
            donors.isnull().mean().rename("donors"),
            donations.isnull().mean().rename("donations"),
        ],
        axis=1,
    ).fillna(0)
    missing = missing[(missing > 0).any(axis=1)]
    fig, ax = plt.subplots(figsize=(10, max(4, len(missing) * 0.4)))
    missing.plot(kind="barh", ax=ax)
    ax.set_title("Missing Value Rate by Column")
    ax.set_xlabel("Proportion Missing")
    return _save(fig, figures_dir / "missing_values.png")


def plot_class_imbalance(dataset: pd.DataFrame, figures_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, target, title in zip(
        axes,
        ["retained_180", "churn_365"],
        ["180-day Retention", "365-day Churn"],
    ):
        counts = dataset[target].dropna().value_counts().sort_index()
        sns.barplot(x=counts.index.astype(int), y=counts.values, ax=ax, palette="Reds")
        ax.set_title(title)
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
    return _save(fig, figures_dir / "class_imbalance.png")


def plot_correlation_heatmap(dataset: pd.DataFrame, figures_dir: Path) -> Path:
    numeric_cols = dataset.select_dtypes(include="number").columns
    corr = dataset[numeric_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, cmap="RdBu_r", center=0, ax=ax, square=False)
    ax.set_title("Feature Correlation Matrix")
    return _save(fig, figures_dir / "correlation_heatmap.png")


def plot_donation_trends(donations: pd.DataFrame, figures_dir: Path) -> Path:
    monthly = (
        donations.set_index("Donation_Date")
        .resample("ME")["Units_Collected"]
        .sum()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=monthly, x="Donation_Date", y="Units_Collected", ax=ax)
    ax.set_title("Monthly Donation Volume Trend")
    return _save(fig, figures_dir / "monthly_donation_trend.png")


def plot_cohort_retention(donations: pd.DataFrame, figures_dir: Path) -> Path:
    donations = donations.copy()
    donations["cohort_month"] = donations.groupby("Donor_ID")["Donation_Date"].transform("min").dt.to_period("M")
    donations["period_number"] = (
        donations["Donation_Date"].dt.to_period("M") - donations["cohort_month"]
    ).apply(lambda value: value.n if pd.notna(value) else None)
    cohort = (
        donations.groupby(["cohort_month", "period_number"])["Donor_ID"]
        .nunique()
        .reset_index(name="active_donors")
    )
    cohort_sizes = donations.groupby("cohort_month")["Donor_ID"].nunique().rename("cohort_size")
    cohort = cohort.merge(cohort_sizes, on="cohort_month")
    cohort["retention_rate"] = cohort["active_donors"] / cohort["cohort_size"]

    pivot = cohort.pivot(index="cohort_month", columns="period_number", values="retention_rate")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot.iloc[:18, :12], cmap="YlGnBu", ax=ax)
    ax.set_title("Donor Cohort Retention Heatmap")
    return _save(fig, figures_dir / "cohort_retention_heatmap.png")


def plot_lifecycle_distribution(dataset: pd.DataFrame, figures_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(dataset["total_donations"], bins=30, ax=axes[0], color="#8B1A1A")
    axes[0].set_title("Distribution of Lifetime Donations")
    sns.histplot(dataset["days_since_last_donation"], bins=30, ax=axes[1], color="#c0392b")
    axes[1].set_title("Days Since Last Donation at Anchor")
    return _save(fig, figures_dir / "donor_lifecycle_distributions.png")


def plot_segment_distribution(rfm: pd.DataFrame, figures_dir: Path) -> Path:
    counts = rfm["segment"].value_counts().reset_index()
    counts.columns = ["segment", "count"]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=counts, x="segment", y="count", ax=ax, palette="Reds_r")
    ax.set_title("RFM Donor Segment Distribution")
    ax.tick_params(axis="x", rotation=20)
    return _save(fig, figures_dir / "rfm_segment_distribution.png")


def run_full_eda(
    donors: pd.DataFrame,
    donations: pd.DataFrame,
    dataset: pd.DataFrame,
    rfm: pd.DataFrame,
    figures_dir: Path,
) -> dict[str, str]:
    """Generate all EDA figures and return path mapping."""
    outputs = {
        "missing_values": str(plot_missing_values(donors, donations, figures_dir)),
        "class_imbalance": str(plot_class_imbalance(dataset, figures_dir)),
        "correlation_heatmap": str(plot_correlation_heatmap(dataset, figures_dir)),
        "monthly_donation_trend": str(plot_donation_trends(donations, figures_dir)),
        "cohort_retention_heatmap": str(plot_cohort_retention(donations, figures_dir)),
        "donor_lifecycle_distributions": str(plot_lifecycle_distribution(dataset, figures_dir)),
        "rfm_segment_distribution": str(plot_segment_distribution(rfm, figures_dir)),
    }
    return outputs
