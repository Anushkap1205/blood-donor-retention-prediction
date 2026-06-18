"""Model training pipeline for donor retention and churn prediction."""

from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import METRICS_DIR, MODELS_DIR, RANDOM_STATE, TEST_SIZE
from src.features.engineering import get_feature_columns


@dataclass
class ModelResult:
    name: str
    target: str
    metrics: dict
    cv_metrics: dict
    model_path: str


def _build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    numeric_features = [c for c in x.columns if c not in {"Gender", "Blood_Group", "age_group"}]
    categorical_features = ["Gender", "Blood_Group", "age_group"]
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )



def _get_models() -> dict:
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier

        models["lightgbm"] = LGBMClassifier(
            n_estimators=300,
            max_depth=-1,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
    except OSError:
        pass
    try:
        from catboost import CatBoostClassifier

        models["catboost"] = CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            auto_class_weights="Balanced",
            random_state=RANDOM_STATE,
            verbose=False,
        )
    except Exception:
        pass
    return models


def train_all_models(
    x: pd.DataFrame,
    y: pd.Series,
    target_name: str,
    groups: pd.Series | None = None,
) -> list[ModelResult]:
    """Train and evaluate all candidate models for a target."""
    if groups is not None:
        from sklearn.model_selection import StratifiedGroupKFold
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        train_idx, test_idx = next(sgkf.split(x, y, groups))
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train = groups.iloc[train_idx]
        groups_test = groups.iloc[test_idx]
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
        groups_train = None
        groups_test = None

    preprocessor = _build_preprocessor(x)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
    }
    
    if groups is not None:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        
    results: list[ModelResult] = []

    for model_name, estimator in _get_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        
        # Apply scale_pos_weight dynamically to XGBoost based on train split imbalance
        if model_name == "xgboost":
            scale_pos = float((y_train == 0).sum() / (y_train == 1).sum())
            estimator.set_params(scale_pos_weight=scale_pos)
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", estimator),
                ]
            )

        # Hyperparameter tuning for all models to ensure fair champion comparison
        tuned_pipeline = pipeline
        from sklearn.model_selection import RandomizedSearchCV
        param_grids = {
            "logistic_regression": {
                "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            },
            "random_forest": {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [5, 10, 15, None],
                "model__min_samples_split": [2, 5, 10],
            },
            "xgboost": {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [3, 5, 7],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__subsample": [0.7, 0.9, 1.0],
            },
            "lightgbm": {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [3, 5, -1],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__subsample": [0.7, 0.9, 1.0],
            },
            "catboost": {
                "model__iterations": [100, 200, 300],
                "model__depth": [4, 6, 8],
                "model__learning_rate": [0.01, 0.05, 0.1],
            }
        }
        if model_name in param_grids:
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_grids[model_name],
                n_iter=5,
                cv=cv,
                scoring="roc_auc",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            search.fit(x_train, y_train, groups=groups_train)
            tuned_pipeline = search.best_estimator_


        cv_scores = cross_validate(
            tuned_pipeline,
            x_train,
            y_train,
            groups=groups_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )
        
        tuned_pipeline.fit(x_train, y_train)
        y_pred = tuned_pipeline.predict(x_test)
        y_prob = tuned_pipeline.predict_proba(x_test)[:, 1]

        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
            brier_score_loss,
        )

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
            "pr_auc": float(average_precision_score(y_test, y_prob)),
            "brier_score": float(brier_score_loss(y_test, y_prob)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "positive_rate_train": float(y_train.mean()),
            "positive_rate_test": float(y_test.mean()),
        }
        cv_metrics = {
            metric: {
                "mean": float(np.mean(cv_scores[f"test_{metric}"])),
                "std": float(np.std(cv_scores[f"test_{metric}"])),
            }
            for metric in scoring
        }

        model_path = MODELS_DIR / f"{target_name}_{model_name}.joblib"
        joblib.dump(tuned_pipeline, model_path)
        metrics_path = METRICS_DIR / f"{target_name}_{model_name}.json"
        with open(metrics_path, "w", encoding="utf-8") as handle:
            json.dump({"metrics": metrics, "cv_metrics": cv_metrics}, handle, indent=2)


        results.append(
            ModelResult(
                name=model_name,
                target=target_name,
                metrics=metrics,
                cv_metrics=cv_metrics,
                model_path=str(model_path),
            )
        )

    leaderboard = sorted(results, key=lambda item: item.metrics["roc_auc"], reverse=True)
    summary_path = METRICS_DIR / f"{target_name}_leaderboard.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(
            [
                {
                    "model": item.name,
                    "roc_auc": item.metrics["roc_auc"],
                    "f1": item.metrics["f1"],
                    "recall": item.metrics["recall"],
                    "model_path": item.model_path,
                }
                for item in leaderboard
            ],
            handle,
            indent=2,
        )
    return leaderboard
