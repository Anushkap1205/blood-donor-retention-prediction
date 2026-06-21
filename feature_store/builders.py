"""Enhanced feature store builders with literature-backed RFMTC+ extensions."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.config import AGE_BINS, AGE_LABELS, CHURN_WINDOW_DAYS, RETENTION_WINDOW_DAYS
from src.features.engineering import build_observation_dataset as _build_base


def _seasonal_features(anchor: pd.Timestamp) -> dict[str, float]:
    month = anchor.month
    quarter = (month - 1) // 3 + 1
    # Indian festival season proxy: Oct-Dec (Diwali/Dussehra) and Mar-Apr
    is_festival_season = int(month in {3, 4, 10, 11, 12})
    return {
        "anchor_month": float(month),
        "anchor_quarter": float(quarter),
        "anchor_sin_month": float(math.sin(2 * math.pi * month / 12)),
        "anchor_cos_month": float(math.cos(2 * math.pi * month / 12)),
        "is_festival_season": is_festival_season,
    }


def _engagement_features(row: dict) -> dict[str, float]:
    total = max(row["total_donations"], 1)
    avg_gap = row.get("avg_gap_days") or 90.0
    std_gap = row.get("std_gap_days") or 0.0
    recency = row.get("days_since_last_donation", 0)
    velocity = row.get("donation_velocity", 0.0)
    recent_score = row.get("recent_activity_score", 0.0)
    camp_ratio = row.get("camp_ratio", 0.0)

    consistency = 1.0 / (1.0 + (std_gap / max(avg_gap, 1.0))) if not np.isnan(std_gap) else 0.0
    # Proxy: camp donors respond to outreach; walk-in donors show self-initiation
    comm_response_proxy = min(1.0, recent_score * 0.6 + camp_ratio * 0.4)
    attendance_proxy = min(1.0, row.get("walkin_ratio", 0.5) * 0.5 + consistency * 0.5)
    loyalty = min(1.0, (total / 10.0) * (1.0 / (1.0 + recency / 180.0)))
    dlv = row.get("total_units_donated", 0.0) * (1.0 + total * 0.1)
    engagement = 0.35 * recent_score + 0.25 * velocity + 0.20 * consistency + 0.20 * camp_ratio

    # NACO 90-day minimum gap guideline proxy
    is_regular_giver = int(56 <= avg_gap <= 120 and total >= 3)

    gap_trend = 0.0
    if row.get("min_gap_days") is not None and not np.isnan(row.get("min_gap_days", np.nan)):
        gap_trend = float(recency - avg_gap)

    return {
        "donation_consistency": float(consistency),
        "preferred_donation_interval_days": float(avg_gap) if not np.isnan(avg_gap) else 90.0,
        "communication_response_rate_proxy": float(comm_response_proxy),
        "campaign_response_history": float(camp_ratio),
        "appointment_attendance_rate_proxy": float(attendance_proxy),
        "engagement_score": float(min(1.0, engagement)),
        "loyalty_score": float(loyalty),
        "donor_lifetime_value": float(dlv),
        "is_regular_giver": is_regular_giver,
        "gap_trend_days": gap_trend,
    }


def enrich_observation_dataset(base: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features on top of point-in-time base dataset."""
    enriched = base.copy()
    extra_rows: list[dict] = []
    for _, row in enriched.iterrows():
        row_dict = row.to_dict()
        seasonal = _seasonal_features(pd.Timestamp(row["anchor_date"]))
        engagement = _engagement_features(row_dict)
        extra_rows.append({**seasonal, **engagement})
    extras = pd.DataFrame(extra_rows, index=enriched.index)
    return pd.concat([enriched, extras], axis=1)


def build_feature_store_dataset(
    donors: pd.DataFrame,
    donations: pd.DataFrame,
    retention_days: int = RETENTION_WINDOW_DAYS,
    churn_days: int = CHURN_WINDOW_DAYS,
) -> pd.DataFrame:
    """Build full feature-store observation dataset."""
    base = _build_base(donors, donations, retention_days, churn_days)
    return enrich_observation_dataset(base)


def get_enhanced_feature_columns() -> list[str]:
    """All model features including engineered extensions."""
    from src.features.engineering import get_feature_columns

    base = get_feature_columns()
    extensions = [
        "anchor_month",
        "anchor_quarter",
        "anchor_sin_month",
        "anchor_cos_month",
        "is_festival_season",
        "donation_consistency",
        "preferred_donation_interval_days",
        "communication_response_rate_proxy",
        "campaign_response_history",
        "appointment_attendance_rate_proxy",
        "engagement_score",
        "loyalty_score",
        "donor_lifetime_value",
        "is_regular_giver",
        "gap_trend_days",
    ]
    return base + extensions


def prepare_enhanced_matrix(
    dataset: pd.DataFrame, target: str
) -> tuple[pd.DataFrame, pd.Series]:
    labeled = dataset.dropna(subset=[target]).copy()
    features = get_enhanced_feature_columns()
    x = labeled[features].copy()
    y = labeled[target].astype(int)
    return x, y
