"""Production inference scorer for donor retention intelligence."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from feature_store.builders import get_enhanced_feature_columns


class DonorScorer:
    """Load champion models and score donors at latest observation."""

    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self.retention_model = None
        self.churn_model = None
        self.feature_columns = get_enhanced_feature_columns()

    def load(self, retention_model_name: str = "logistic_regression", churn_model_name: str = "catboost") -> None:
        self.retention_model = joblib.load(
            self.models_dir / f"retained_180_{retention_model_name}.joblib"
        )
        self.churn_model = joblib.load(
            self.models_dir / f"churn_365_{churn_model_name}.joblib"
        )

    def score_latest(self, dataset: pd.DataFrame) -> pd.DataFrame:
        latest = (
            dataset.sort_values("anchor_date")
            .groupby("Donor_ID", as_index=False)
            .tail(1)
            .copy()
        )
        x = latest[self.feature_columns]
        latest["retention_probability"] = self.retention_model.predict_proba(x)[:, 1]
        latest["churn_probability"] = self.churn_model.predict_proba(x)[:, 1]
        latest["churn_risk_score"] = latest["churn_probability"]
        return latest
