# Blood Donor Retention Prediction — Final Research Report

## Executive Summary

This project implements a research-guided machine learning system to predict blood donor retention (180-day return) and churn (365-day non-return) using the Samarpan Blood Bank synthetic dataset (3,500 donors, 24,195 donations, 2023–2025). The pipeline follows RFMTC principles (Yeh et al., 2009), contemporary ML benchmarks (Kauten et al., 2021; Liu et al., 2022), and India-specific retention evidence (Mohammed 2025; Malhotra 2026).

## Target Construction

### Problem 1: Retention (`retained_180 = 1`)

At each donation anchor date *t*, label = 1 if another donation occurs within (*t*, *t* + 180 days].

**Justification:** 180 days aligns with Indian whole-blood eligibility (≈90-day minimum gap × 2) and operational campaign cycles. Matches claude.md specification and Liu et al. (2022) interval-based framing.

**Censoring:** Observations where *t* + 180 > max(dataset date) are excluded.

### Problem 2: Churn (`churn_365 = 1`)

At anchor *t*, label = 1 if **no** donation occurs within (*t*, *t* + 365 days].

**Justification:** One-year inactivity is standard lapse definition (Yang et al., 2020; van Dongen, 2015). Enables long-horizon reactivation campaigns.

## Dataset & EDA Highlights

- **Class imbalance (retention):** ~70% positive (retained) — consistent with UCI transfusion dataset (~76% non-return inverted at longer windows).
- **Missing values:** Only `Camp_ID` (walk-ins); handled structurally.
- **Temporal trend:** Stable monthly collections with mild seasonality (see `outputs/figures/monthly_donation_trend.png`).
- **Cohort analysis:** Retention heatmap in `outputs/figures/cohort_retention_heatmap.png`.

## Model Results

### 180-Day Retention

| Model | ROC-AUC | Recall | F1 | PR-AUC |
|-------|---------|--------|-----|--------|
| **Random Forest** (selected) | **0.605** | 0.761 | 0.758 | 0.758 |
| XGBoost | 0.601 | **0.943** | 0.808 | — |
| Logistic Regression | 0.591 | 0.643 | 0.695 | — |

**Selection:** Random Forest chosen for best ROC-AUC and balanced precision-recall on held-out test set. **Operational note:** If maximizing at-risk donor capture, XGBoost recall (0.943) may be preferred despite lower AUC — consistent with Kiarie et al. (2024) emphasis on recall for imbalanced retention.

### 365-Day Churn

| Model | ROC-AUC | Recall | F1 |
|-------|---------|--------|-----|
| **Random Forest** (selected) | **0.680** | 0.518 | 0.287 |
| CatBoost | 0.675 | **0.612** | 0.299 |
| Logistic Regression | 0.671 | 0.661 | 0.279 |

**Selection:** Random Forest for best discrimination (AUC). Churn is minority class (~15%); PR-AUC and recall should guide threshold tuning in production.

5-fold CV ROC-AUC stable (σ ≈ 0.01). Full metrics: `outputs/metrics/`.

## Explainability

**Top retention drivers (SHAP, Random Forest):**

1. `walkin_donation_count` / `walkin_ratio`
2. `camp_ratio`
3. `min_gap_days`, `days_since_last_donation`
4. `total_donations`, `tenure_days`

**Literature comparison:**

| Agrees | Differs |
|--------|---------|
| Recency and frequency in top tier (Yeh 2009; Liu 2022) | Channel mix (camp/walk-in) dominates here — India camp context (Mohammed 2025) |
| Tenure and donation count matter (Kauten 2021) | Modest AUC suggests synthetic data limits behavioral signal |
| Gap statistics capture habit (Almutairi 2019) | Blood group weaker than literature — limited variance in synthetic set |

## Segmentation

RFM segments: 1,241 Active, 874 Lost, 859 At-Risk, 492 Loyal. See `reports/segmentation_report.md`.

## Recommendations

1. Deploy weekly scoring → `reports/donor_action_plan.csv`
2. Prioritize SMS for medium/high risk; phone for >12-month inactive (Yang 2020)
3. Camp-targeted outreach for high `camp_ratio` donors
4. First-donation counseling within 7 days (Bagot 2016)
5. Recalibrate thresholds on real Samarpan data when available

## Reproducibility

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py
```

Outputs: models, figures, metrics, and reports under `outputs/` and `reports/`.
