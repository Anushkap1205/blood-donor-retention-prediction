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

### 180-Day Retention (Base Rate: 70.11%)
*Evaluation metrics computed on the group-held-out test set:*

| Model | ROC-AUC | PR-AUC | F1 | Recall | Precision | Brier Score |
|-------|---------|--------|----|--------|-----------|-------------|
| **Logistic Regression** (selected) | **0.6130** | **0.7622** | 0.7228 | 0.6807 | 0.7705 | 0.2370 |
| Random Forest | 0.6122 | 0.7559 | 0.7703 | 0.7740 | 0.7665 | **0.2246** |
| LightGBM | 0.6093 | 0.7500 | **0.7811** | **0.7954** | 0.7672 | 0.2375 |
| CatBoost | 0.6080 | 0.7493 | **0.7811** | **0.7954** | 0.7672 | 0.2374 |
| XGBoost | 0.6061 | 0.7463 | 0.7810 | 0.7944 | **0.7680** | 0.2375 |

**Selection:** Logistic Regression chosen for best overall ROC-AUC and PR-AUC. However, for operational campaigns prioritizing capture rate, tree models like LightGBM / CatBoost may be preferred for their higher recall (79.54%).

### 365-Day Churn (Base Rate: 10.22%)
*Evaluation metrics computed on the group-held-out test set:*

| Model | ROC-AUC | PR-AUC | F1 | Recall | Precision | Brier Score |
|-------|---------|--------|----|--------|-----------|-------------|
| **CatBoost** (selected) | **0.7007** | 0.1880 | **0.3226** | 0.6635 | **0.2131** | 0.2217 |
| Logistic Regression | 0.6988 | **0.1926** | 0.3005 | **0.7013** | 0.1913 | 0.2218 |
| XGBoost | 0.6945 | 0.1800 | 0.3223 | 0.6572 | 0.2135 | 0.2196 |
| Random Forest | 0.6941 | 0.1736 | 0.2948 | 0.4937 | 0.2102 | **0.1516** |
| LightGBM | 0.6865 | 0.1789 | 0.3223 | 0.6572 | 0.2135 | 0.2207 |

**Selection:** CatBoost chosen for best discrimination (ROC-AUC) and balanced F1 score under class imbalance. Under extreme target imbalance (10.22%), PR-AUC is the primary performance index; Logistic Regression has the highest PR-AUC (0.1926) and Recall (0.7013) but suffers on precision.

5-fold group cross-validation ROC-AUC remains stable (σ ≈ 0.01). Reliability calibration curves and curves plots saved under `outputs/figures/`.

## Explainability

**Top Retention Drivers (Permutation Importance, Logistic Regression):**
1. `recent_activity_score`
2. `donations_last_6_months`
3. `walkin_ratio` / `camp_ratio`
4. `max_gap_days`

**Top Churn Drivers (SHAP Summary, CatBoost):**
1. `walkin_donation_count`
2. `walkin_ratio` / `camp_ratio`
3. `recent_activity_score`
4. `std_gap_days`

**Literature Comparison & Performance Gap:**

| Agrees | Differs |
|--------|---------|
| Recency and frequency in top tier (Yeh 2009; Liu 2022) | Channel mix (camp vs walk-in) is the dominant predictor cohort here. |
| Tenure and gap metrics capture habit (van Dongen 2015) | Low overall ROC-AUC (0.61–0.70) vs. literature benchmarks (0.80+) highlights synthetic dataset limitations. |

*Why the Model Performance Gap?*
Literature benchmarks achieving AUCs above 0.80+ typically incorporate deep behavioral records (e.g., historical deferral details, precise travel distances, SMS response logs, hemoglobin levels, and marketing campaign histories). In our synthetic dataset, these signals are absent, and the simulated generator adds uniform noise, bounding the predictive power of RFMTC constructs.

## Segmentation

RFM segments: 1,241 Active, 874 Lost, 859 At-Risk, 492 Loyal. See `reports/segmentation_report.md`.

## Recommendations

> [!WARNING]
> **Synthetic Data Caveat:**
> All models, segmentations, and rules are developed and validated against the simulated Samarpan V2 dataset. These patterns must be validated against real, anonymized hospital/blood bank data before deploying in a live clinical workflow.

1. Deploy weekly scoring using `reports/donor_action_plan.csv` for targeted communications.
2. Prioritize SMS for medium/high risk; phone outreach for >12-month inactive (Yang 2020).
3. Mobile camp-targeted promotions for donors with a high historical `camp_ratio`.
4. First-donation counseling within 7 days (Bagot 2016).
5. Recalibrate thresholds on real Samarpan clinical logs when available.

## Reproducibility

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py
```

Outputs: models, figures, metrics, and reports under `outputs/` and `reports/`.

