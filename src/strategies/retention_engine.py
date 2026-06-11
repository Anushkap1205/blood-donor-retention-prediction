"""Retention strategy recommendations from model scores and RFM segments."""

from __future__ import annotations

import pandas as pd


INTERVENTION_LIBRARY = [
    {
        "segment": "High churn risk",
        "condition": "retention_probability < 0.50",
        "intervention": "SMS reminder with altruistic appeal",
        "evidence": "Yang et al. (2020); Mohammed et al. (2025, South India)",
        "expected_impact": "High",
        "priority": 1,
    },
    {
        "segment": "High churn risk",
        "condition": "retention_probability < 0.50 and months_since_last_donation > 12",
        "intervention": "Personalized phone outreach",
        "evidence": "Yang et al. (2020) RCT: phone calls effective for long-inactive donors",
        "expected_impact": "High",
        "priority": 2,
    },
    {
        "segment": "Camp-oriented donors",
        "condition": "camp_ratio >= 0.60",
        "intervention": "Local camp notifications and early registration",
        "evidence": "Srivastava et al. (2025, South India); van Dongen (2015)",
        "expected_impact": "Medium-High",
        "priority": 3,
    },
    {
        "segment": "First-time donors",
        "condition": "is_first_donation == 1",
        "intervention": "Post-donation counseling and thank-you follow-up within 7 days",
        "evidence": "Bagot et al. (2016); Masser et al. (2012, TPB)",
        "expected_impact": "High",
        "priority": 1,
    },
    {
        "segment": "Frequent donors",
        "condition": "total_donations >= 5 and retention_probability >= 0.80",
        "intervention": "Recognition certificate and loyalty appreciation campaign",
        "evidence": "Malhotra et al. (2026, North India); Grazzini (2021)",
        "expected_impact": "Medium",
        "priority": 4,
    },
    {
        "segment": "At-risk donors",
        "condition": "0.50 <= retention_probability < 0.80",
        "intervention": "Personalized donation invitation with preferred channel",
        "evidence": "Kauten et al. (2021); Liu et al. (2022)",
        "expected_impact": "Medium",
        "priority": 3,
    },
    {
        "segment": "Female donors",
        "condition": "Gender == Female and retention_probability < 0.70",
        "intervention": "Deferral-aware SMS with hemoglobin education and flexible scheduling",
        "evidence": "Marwaha et al. (2012); Sachdeva et al. (2023, India)",
        "expected_impact": "Medium",
        "priority": 3,
    },
]


def risk_category(probability: float) -> str:
    if probability >= 0.80:
        return "High Retention"
    if probability >= 0.50:
        return "Medium Risk"
    return "High Churn Risk"


def recommend_interventions(row: pd.Series) -> list[str]:
    """Return ranked interventions for a donor snapshot."""
    recommendations: list[tuple[int, str]] = []
    probability = row.get("retention_probability", 0.5)

    if probability < 0.50:
        recommendations.append((1, "SMS reminder with altruistic appeal"))
        if row.get("days_since_last_donation", 0) > 365:
            recommendations.append((2, "Personalized phone outreach"))
    elif probability < 0.80:
        recommendations.append((3, "Personalized donation invitation"))

    if row.get("camp_ratio", 0) >= 0.60:
        recommendations.append((3, "Notify about nearby donation camps"))

    if row.get("is_first_donation", 0) == 1:
        recommendations.append((1, "Post-donation counseling and thank-you follow-up"))

    if row.get("total_donations", 0) >= 5 and probability >= 0.80:
        recommendations.append((4, "Recognition certificate and loyalty appreciation"))

    if row.get("Gender") == "Female" and probability < 0.70:
        recommendations.append((3, "Deferral-aware SMS with flexible scheduling"))

    if not recommendations:
        recommendations.append((5, "Maintain standard engagement cadence"))

    recommendations.sort(key=lambda item: item[0])
    return [text for _, text in recommendations]


def build_action_plan(scored_donors: pd.DataFrame) -> pd.DataFrame:
    """Attach risk categories and intervention lists to scored donors."""
    plan = scored_donors.copy()
    plan["risk_category"] = plan["retention_probability"].map(risk_category)
    plan["recommended_interventions"] = plan.apply(recommend_interventions, axis=1)
    return plan


def intervention_ranking() -> pd.DataFrame:
    """Static intervention ranking table for reporting."""
    return pd.DataFrame(INTERVENTION_LIBRARY).sort_values("priority")
