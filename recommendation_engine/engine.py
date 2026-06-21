"""Evidence-backed retention recommendation engine with rules and causal uplift."""

from __future__ import annotations

import yaml
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.strategies.retention_engine import build_action_plan, risk_category
from training.uplift import UpliftRecommendation, recommend_causal_intervention


@dataclass
class DonorRecommendation:
    donor_id: str
    retention_probability: float
    churn_probability: float
    churn_risk: str
    next_donation_estimate_days: float | None
    primary_intervention: str
    secondary_intervention: str | None
    causal_intervention: str
    expected_uplift: float
    confidence: float
    explanation_summary: str


def _estimate_next_donation(row: pd.Series) -> float | None:
    avg_gap = row.get("preferred_donation_interval_days") or row.get("avg_gap_days")
    if pd.isna(avg_gap):
        return None
    recency = row.get("days_since_last_donation", 0)
    return max(0.0, float(avg_gap) - recency)


def _rule_based_intervention(row: pd.Series) -> tuple[str, str | None]:
    """Research-backed rules engine (extends src/strategies/retention_engine)."""
    from src.strategies.retention_engine import recommend_interventions

    primary = recommend_interventions(row)
    primary_action = primary[0] if primary else "Maintain standard engagement cadence"
    secondary = None

    churn_prob = row.get("churn_probability", 0.0)
    ret_prob = row.get("retention_probability", 1.0)
    missed_reminders = row.get("communication_response_rate_proxy", 0.5) < 0.3

    if churn_prob > 0.80:
        return "Schedule personal outreach (high churn risk)", "Escalate to donor counselor"
    if missed_reminders and row.get("camp_ratio", 0) < 0.5:
        return "Switch communication channel (SMS → WhatsApp or phone)", primary_action
    if row.get("communication_response_rate_proxy", 0) > 0.6:
        return primary_action, "Prioritize SMS channel"
    if ret_prob >= 0.80 and row.get("total_donations", 0) >= 5:
        return "Invite to ambassador program", "Recognition certificate"
    return primary_action, secondary


def generate_recommendations(scored_donors: pd.DataFrame) -> pd.DataFrame:
    """Full recommendation output merging rules + causal uplift."""
    records = []
    for _, row in scored_donors.iterrows():
        primary, secondary = _rule_based_intervention(row)
        baseline = float(row.get("retention_probability", 0.5))
        causal: UpliftRecommendation = recommend_causal_intervention(row, baseline)
        next_days = _estimate_next_donation(row)

        records.append(
            DonorRecommendation(
                donor_id=str(row["Donor_ID"]),
                retention_probability=baseline,
                churn_probability=float(row.get("churn_probability", 0.0)),
                churn_risk=risk_category(baseline),
                next_donation_estimate_days=next_days,
                primary_intervention=primary,
                secondary_intervention=secondary,
                causal_intervention=causal.recommended_intervention,
                expected_uplift=causal.expected_retention_improvement,
                confidence=causal.confidence_score,
                explanation_summary="; ".join(
                    filter(
                        None,
                        [
                            f"Retention P={baseline:.2f}",
                            f"Churn P={row.get('churn_probability', 0):.2f}",
                            f"Next donation ~{int(next_days)}d" if next_days is not None else None,
                        ],
                    )
                ),
            ).__dict__
        )
    return pd.DataFrame(records)


def build_full_action_plan(scored_donors: pd.DataFrame) -> pd.DataFrame:
    """Merge legacy action plan with enhanced recommendations."""
    base = build_action_plan(scored_donors)
    enhanced = generate_recommendations(scored_donors)
    return base.merge(
        enhanced[
            [
                "donor_id",
                "primary_intervention",
                "secondary_intervention",
                "causal_intervention",
                "expected_uplift",
                "confidence",
                "next_donation_estimate_days",
                "explanation_summary",
            ]
        ],
        left_on="Donor_ID",
        right_on="donor_id",
        how="left",
    ).drop(columns=["donor_id"])
