"""Unit tests for donor retention intelligence system."""

from __future__ import annotations

import pandas as pd
import pytest

from feature_store.builders import enrich_observation_dataset, get_enhanced_feature_columns
from src.data.cleaning import clean_donations, clean_donors
from src.data.loader import load_donor_donation_tables
from src.features.engineering import build_observation_dataset
from recommendation_engine.engine import generate_recommendations


@pytest.fixture(scope="module")
def sample_data():
    donors, donations = load_donor_donation_tables()
    donors = clean_donors(donors)
    donations = clean_donations(donations)
    return donors, donations


def test_observation_dataset_no_future_leakage(sample_data):
    donors, donations = sample_data
    dataset = build_observation_dataset(donors, donations)
    assert "Donor_Status_Label" not in get_enhanced_feature_columns()
    assert "Donation_Frequency_Label" not in get_enhanced_feature_columns()
    assert dataset["total_donations"].min() >= 1


def test_enhanced_features(sample_data):
    donors, donations = sample_data
    base = build_observation_dataset(donors, donations)
    enriched = enrich_observation_dataset(base)
    for col in ["engagement_score", "loyalty_score", "donation_consistency"]:
        assert col in enriched.columns
        assert enriched[col].between(0, enriched[col].max()).all()


def test_recommendation_engine_outputs(sample_data):
    donors, donations = sample_data
    dataset = build_observation_dataset(donors, donations)
    latest = dataset.sort_values("anchor_date").groupby("Donor_ID").tail(1).copy()
    latest["retention_probability"] = 0.6
    latest["churn_probability"] = 0.2
    recs = generate_recommendations(latest.head(20))
    assert "primary_intervention" in recs.columns
    assert "causal_intervention" in recs.columns
    assert len(recs) == 20


def test_point_in_time_labels(sample_data):
    donors, donations = sample_data
    dataset = build_observation_dataset(donors, donations)
    labeled = dataset.dropna(subset=["retained_180"])
    assert labeled["retained_180"].isin([0, 1]).all()
