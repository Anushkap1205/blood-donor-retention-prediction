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
    """
    Determine risk tier based on calibrated 180-day return probability.
    
    Threshold Rationale:
    - High Retention (prob >= 0.80): Identifies the top 30% of secure, habituated donors
      ideal for milestone recognition.
    - Medium Risk (0.50 <= prob < 0.80): Below-average return chance (base rate ~70.11%),
      targeted with soft, channel-preferred invites.
    - High Churn Risk (prob < 0.50): Crucial drop below 50% chance of return, requiring
      direct intervention alerts.
    """
    if probability >= 0.80:
        return "High Retention"
    if probability >= 0.50:
        return "Medium Risk"
    return "High Churn Risk"


def recommend_interventions(row: pd.Series) -> list[str]:
    """
    Return a ranked, mutually exclusive list of interventions for a donor.
    
    Rule Precedence Order (Single Campaign Contact Policy):
    To prevent communication fatigue, each donor is assigned exactly one primary action:
    1. First-Time Donor Cold Start -> Counsel/Follow-up (Bypasses predictive models entirely)
    2. Long-Term Lapsed -> Reactivation Phone Call (Bypasses SMS to avoid redundant channels)
    3. High Churn Risk (Recent) -> Altruistic SMS (Merged with Camp details if camp-oriented)
    4. Sub-optimal Female Donor -> Deferral-Aware Iron/Scheduling SMS
    5. Camp-Oriented Donor -> Geocoded Camp Notification
    6. Loyal Veteran -> Appreciation Certificate
    7. Medium Risk -> Standard Invite
    """
    is_first = row.get("is_first_donation", 0) == 1
    if is_first:
        # COLD-START ROUTE: Bypass model scoring. Route directly to counseling.
        return ["Post-donation counseling and thank-you follow-up"]

    days_inactive = row.get("days_since_last_donation", 0)
    churn_prob = row.get("churn_probability", 0.0)
    ret_prob = row.get("retention_probability", 1.0)
    gender = row.get("Gender", "Male")
    camp_ratio = row.get("camp_ratio", 0.0)
    total_donations = row.get("total_donations", 0)

    # 1. PRIORITY 1: Long-Term Reactivation Call (Inactive > 365 Days)
    # Trigger: Churn probability >= 0.50 (Nearly 5x the 10.22% churn base rate)
    if churn_prob >= 0.50 and days_inactive > 365:
        return ["Personalized phone outreach"]

    # 2. PRIORITY 2: Altruistic Churn SMS (Active but High Risk)
    # Trigger: Churn probability >= 0.50
    if churn_prob >= 0.50:
        # If donor is camp-oriented, adapt the SMS context rather than sending two notifications
        if camp_ratio >= 0.60:
            return ["Altruistic SMS reminder with camp-location matching"]
        return ["SMS reminder with altruistic appeal"]

    # 3. PRIORITY 3: Deferral-Aware iron/scheduling SMS
    # Trigger: Gender == 'Female' and retention_prob < 0.70
    # Reframing Protected Attribute: Gender is used here temporarily as a proxy for physical
    # deferral risk (specifically anemia as per Marwaha 2012) due to database constraints in V2.
    # ROADMAP: Replace with a dedicated deferral-risk classifier once clinical logs are available.
    if gender == "Female" and ret_prob < 0.70:
        return ["Deferral-aware SMS with flexible scheduling"]

    # 4. PRIORITY 4: Camp-Targeted notifications (Non-high-risk camp-goers)
    if camp_ratio >= 0.60:
        return ["Notify about nearby donation camps"]

    # 5. PRIORITY 5: Milestone Appreciation for Veteran Cohorts
    if total_donations >= 5 and ret_prob >= 0.80:
        return ["Recognition certificate and loyalty appreciation"]

    # 6. PRIORITY 6: Personalized Donation Invitation for Medium-Risk Donors
    if ret_prob < 0.80:
        return ["Personalized donation invitation"]

    # 7. DEFAULT: Standard cadence
    return ["Maintain standard engagement cadence"]


def build_action_plan(scored_donors: pd.DataFrame) -> pd.DataFrame:
    """Attach risk categories and intervention lists to scored donors."""
    plan = scored_donors.copy()
    plan["risk_category"] = plan["retention_probability"].map(risk_category)
    
    # Override risk category for first-time cold starts
    if "is_first_donation" in plan.columns:
        plan.loc[plan["is_first_donation"] == 1, "risk_category"] = "First-Time Cold Start"
        
    plan["recommended_interventions"] = plan.apply(recommend_interventions, axis=1)
    return plan


def intervention_ranking() -> pd.DataFrame:
    """Static intervention ranking table for reporting."""
    return pd.DataFrame(INTERVENTION_LIBRARY).sort_values("priority")

