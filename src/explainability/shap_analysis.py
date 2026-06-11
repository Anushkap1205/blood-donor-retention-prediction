"""SHAP and permutation importance for model explainability."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance


def _transform_features(pipeline, x: pd.DataFrame) -> np.ndarray:
    return pipeline.named_steps["preprocessor"].transform(x)


def _feature_names(pipeline, x: pd.DataFrame) -> list[str]:
    preprocessor = pipeline.named_steps["preprocessor"]
    return list(preprocessor.get_feature_names_out())


def run_permutation_importance(
    pipeline,
    x: pd.DataFrame,
    y: pd.Series,
    target_name: str,
    model_name: str,
    figures_dir: Path,
    metrics_dir: Path,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute permutation importance on holdout-style sample."""
    sample = x.sample(min(5000, len(x)), random_state=random_state)
    y_sample = y.loc[sample.index]
    x_transformed = _transform_features(pipeline, sample)
    result = permutation_importance(
        pipeline.named_steps["model"],
        x_transformed,
        y_sample,
        n_repeats=10,
        random_state=random_state,
        n_jobs=-1,
    )
    names = _feature_names(pipeline, sample)
    importance = (
        pd.DataFrame(
            {
                "feature": names,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    top = importance.head(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"][::-1], top["importance_mean"][::-1])
    ax.set_title(f"{target_name} / {model_name}: Permutation Importance")
    fig.tight_layout()
    fig.savefig(figures_dir / f"{target_name}_{model_name}_permutation_importance.png", dpi=180)
    plt.close(fig)

    output = metrics_dir / f"{target_name}_{model_name}_permutation_importance.json"
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(importance.head(30).to_dict(orient="records"), handle, indent=2)
    return importance


def run_shap_summary(
    pipeline,
    x: pd.DataFrame,
    target_name: str,
    model_name: str,
    figures_dir: Path,
    metrics_dir: Path,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate SHAP summary for tree-based models when supported."""
    model = pipeline.named_steps["model"]
    sample = x.sample(min(500, len(x)), random_state=random_state)
    x_transformed = _transform_features(pipeline, sample)
    feature_names = _feature_names(pipeline, sample)

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_transformed)
    except Exception:
        explainer = shap.Explainer(
            lambda data: model.predict_proba(data)[:, 1],
            x_transformed,
            feature_names=feature_names,
        )
        shap_values = explainer(x_transformed)

    if isinstance(shap_values, list):
        values = np.asarray(shap_values[1])
    elif hasattr(shap_values, "values"):
        values = np.asarray(shap_values.values)
        if values.ndim == 3:
            values = values[:, :, 1]
    else:
        values = np.asarray(shap_values)
        if values.ndim == 3:
            values = values[:, :, 1]

    mean_abs = np.abs(values).mean(axis=0).ravel()
    importance = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    shap.summary_plot(
        values,
        features=x_transformed,
        feature_names=feature_names,
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    plt.savefig(figures_dir / f"{target_name}_{model_name}_shap_summary.png", dpi=180, bbox_inches="tight")
    plt.close()

    output = metrics_dir / f"{target_name}_{model_name}_shap_importance.json"
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(importance.head(30).to_dict(orient="records"), handle, indent=2)
    return importance
