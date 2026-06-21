"""Streamlit dashboard — Executive, Donor, and Operations views."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(
    page_title="Samarpan Donor Retention Intelligence",
    page_icon="🩸",
    layout="wide",
)


@st.cache_data
def load_data():
    dataset_path = PROJECT_ROOT / "feature_store" / "artifacts" / "observation_dataset.parquet"
    action_path = PROJECT_ROOT / "reports" / "donor_action_plan.csv"
    metrics_path = PROJECT_ROOT / "outputs" / "metrics" / "pipeline_summary_v2.json"

    if dataset_path.exists():
        dataset = pd.read_parquet(dataset_path)
    else:
        csv_fallback = PROJECT_ROOT / "outputs" / "modeling_dataset.csv"
        dataset = pd.read_csv(csv_fallback, parse_dates=["anchor_date"]) if csv_fallback.exists() else pd.DataFrame()

    action_plan = pd.read_csv(action_path) if action_path.exists() else pd.DataFrame()
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as handle:
            metrics = json.load(handle)
    return dataset, action_plan, metrics


def executive_view(dataset: pd.DataFrame, action_plan: pd.DataFrame):
    st.header("Executive View")
    col1, col2, col3, col4 = st.columns(4)

    if not dataset.empty and "retained_180" in dataset.columns:
        retention_rate = dataset["retained_180"].dropna().mean()
        col1.metric("180-Day Retention Rate", f"{retention_rate:.1%}")
    if not action_plan.empty and "churn_probability" in action_plan.columns:
        high_risk = (action_plan["churn_probability"] >= 0.5).sum()
        col2.metric("High-Risk Donors", f"{high_risk:,}")
        col3.metric("Donors Scored", f"{len(action_plan):,}")
    if not dataset.empty:
        col4.metric("Observation Events", f"{len(dataset):,}")

    if not action_plan.empty and "risk_category" in action_plan.columns:
        fig = px.pie(action_plan, names="risk_category", title="Donor Risk Distribution")
        st.plotly_chart(fig, use_container_width=True)

    if not action_plan.empty and "primary_intervention" in action_plan.columns:
        intervention_counts = action_plan["primary_intervention"].value_counts().head(10)
        fig2 = px.bar(x=intervention_counts.index, y=intervention_counts.values, title="Top Recommended Interventions")
        st.plotly_chart(fig2, use_container_width=True)


def donor_view(action_plan: pd.DataFrame, dataset: pd.DataFrame):
    st.header("Donor View")
    if action_plan.empty:
        st.warning("Run the production pipeline to generate donor scores.")
        return

    donor_ids = sorted(action_plan["Donor_ID"].astype(str).unique())
    selected = st.selectbox("Select Donor", donor_ids)
    row = action_plan[action_plan["Donor_ID"].astype(str) == selected].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Retention Probability", f"{row.get('retention_probability', 0):.1%}")
    c2.metric("Churn Probability", f"{row.get('churn_probability', 0):.1%}")
    c3.metric("Risk Category", row.get("risk_category", "N/A"))

    st.subheader("Recommended Actions")
    st.write(f"**Primary:** {row.get('primary_intervention', row.get('recommended_interventions', 'N/A'))}")
    if "causal_intervention" in row:
        st.write(f"**Causal (uplift):** {row['causal_intervention']} (+{row.get('expected_uplift', 0):.1%} expected)")
    if "next_donation_estimate_days" in row and pd.notna(row["next_donation_estimate_days"]):
        st.write(f"**Estimated days to next donation:** {int(row['next_donation_estimate_days'])}")

    expl_path = PROJECT_ROOT / "reports" / "donor_explanations.json"
    if expl_path.exists():
        with open(expl_path, encoding="utf-8") as handle:
            explanations = json.load(handle)
        match = next((e for e in explanations if str(e.get("Donor_ID")) == selected), None)
        if match:
            st.subheader("SHAP Explanation")
            st.write(f"Risk level: **{match['risk_level']}**")
            for reason in match.get("narrative_reasons", []):
                st.write(f"- {reason}")


def operations_view(dataset: pd.DataFrame, action_plan: pd.DataFrame):
    st.header("Operations View")
    if dataset.empty:
        st.info("No dataset loaded.")
        return

    if "anchor_date" in dataset.columns:
        dataset = dataset.copy()
        dataset["anchor_date"] = pd.to_datetime(dataset["anchor_date"])
        monthly = (
            dataset.groupby(dataset["anchor_date"].dt.to_period("M"))
            .agg(donations=("Donor_ID", "count"), retention=("retained_180", "mean"))
            .reset_index()
        )
        monthly["anchor_date"] = monthly["anchor_date"].astype(str)
        fig = px.line(monthly, x="anchor_date", y="donations", title="Monthly Donation Volume")
        st.plotly_chart(fig, use_container_width=True)
        if "retention" in monthly.columns:
            fig2 = px.line(monthly, x="anchor_date", y="retention", title="Retention Trend")
            st.plotly_chart(fig2, use_container_width=True)

    if "Blood_Group" in dataset.columns:
        bg = dataset.groupby("Blood_Group").size().reset_index(name="count")
        fig3 = px.bar(bg, x="Blood_Group", y="count", title="Donations by Blood Group")
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Campaign Planning — High Churn Risk Queue")
    if not action_plan.empty and "churn_probability" in action_plan.columns:
        queue = action_plan[action_plan["churn_probability"] >= 0.3].sort_values(
            "churn_probability", ascending=False
        ).head(50)
        st.dataframe(
            queue[
                [c for c in ["Donor_ID", "churn_probability", "retention_probability", "primary_intervention", "causal_intervention"] if c in queue.columns]
            ],
            use_container_width=True,
        )


def main():
    st.title("Samarpan Blood Bank — Donor Retention Intelligence")
    st.caption("Production dashboard | Synthetic dataset — validate on real records before deployment")

    dataset, action_plan, metrics = load_data()

    tab1, tab2, tab3 = st.tabs(["Executive", "Donor Profile", "Operations"])
    with tab1:
        executive_view(dataset, action_plan)
    with tab2:
        donor_view(action_plan, dataset)
    with tab3:
        operations_view(dataset, action_plan)


if __name__ == "__main__":
    main()
