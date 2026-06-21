"""Local SHAP explanations for individual donors."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap


def _transform(pipeline, x: pd.DataFrame):
    return pipeline.named_steps["preprocessor"].transform(x)


def _feature_names(pipeline, x: pd.DataFrame) -> list[str]:
    return list(pipeline.named_steps["preprocessor"].get_feature_names_out())


def explain_donor(
    pipeline,
    donor_row: pd.DataFrame,
    top_k: int = 5,
) -> dict:
    """Generate human-readable local explanation for one donor."""
    model = pipeline.named_steps["model"]
    x_transformed = _transform(pipeline, donor_row)
    names = _feature_names(pipeline, donor_row)

    try:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(x_transformed)
        if isinstance(values, list):
            values = values[1]
    except Exception:
        explainer = shap.Explainer(
            lambda data: model.predict_proba(data)[:, 1],
            x_transformed,
            feature_names=names,
        )
        sv = explainer(x_transformed)
        values = sv.values if hasattr(sv, "values") else sv

    if values.ndim == 3:
        values = values[:, :, 1]
    values = np.asarray(values).ravel()

    importance = pd.DataFrame({"feature": names, "shap_value": values})
    importance["abs_shap"] = importance["shap_value"].abs()
    top = importance.sort_values("abs_shap", ascending=False).head(top_k)

    prob = float(pipeline.predict_proba(donor_row)[:, 1][0])
    risk_level = "high" if prob < 0.5 else "medium" if prob < 0.8 else "low"

    reasons = []
    for _, row in top.iterrows():
        direction = "increases" if row["shap_value"] > 0 else "decreases"
        reasons.append(f"{row['feature']} {direction} retention probability (SHAP={row['shap_value']:.3f})")

    narrative_parts = []
    raw = donor_row.iloc[0]
    if raw.get("days_since_last_donation", 0) > 180:
        narrative_parts.append(
            f"Last donation was {int(raw['days_since_last_donation'])} days ago"
        )
    if raw.get("communication_response_rate_proxy", 1) < 0.4:
        narrative_parts.append("Communication response proxy is low")
    if raw.get("appointment_attendance_rate_proxy", 1) < 0.5:
        narrative_parts.append("Appointment attendance proxy suggests no-show risk")
    if raw.get("is_first_donation", 0) == 1:
        narrative_parts.append("First-time donor with no established donation habit")

    return {
        "retention_probability": prob,
        "risk_level": risk_level,
        "top_shap_features": top.to_dict(orient="records"),
        "narrative_reasons": narrative_parts or reasons[:3],
        "full_shap_reasons": reasons,
    }


def batch_explain(
    model_path: Path,
    donors: pd.DataFrame,
    feature_columns: list[str],
    output_path: Path,
    sample_size: int = 100,
) -> pd.DataFrame:
    """Explain a sample of donors and save JSON explanations."""
    pipeline = joblib.load(model_path)
    sample = donors.sample(min(sample_size, len(donors)), random_state=42)
    explanations = []
    for _, row in sample.iterrows():
        x = row[feature_columns].to_frame().T
        exp = explain_donor(pipeline, x)
        exp["Donor_ID"] = row.get("Donor_ID", "")
        explanations.append(exp)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(explanations, handle, indent=2, default=str)
    return pd.DataFrame([{"Donor_ID": e["Donor_ID"], "risk_level": e["risk_level"]} for e in explanations])
