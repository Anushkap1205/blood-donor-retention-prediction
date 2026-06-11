# Retention Strategy Recommendations

Evidence-ranked interventions mapped to model outputs. See also `reports/blood_donor_retention_strategies_india.md` for India-specific detail.

## Risk Tiers (from `retention_probability`)

| Tier | Probability | Action |
|------|-------------|--------|
| High Retention | ≥ 0.80 | Maintain cadence; loyalty recognition if frequent |
| Medium Risk | 0.50 – 0.79 | Personalized invitation; camp alerts if camp-oriented |
| High Churn Risk | < 0.50 | SMS reminder; phone if >365 days since last donation |

## Intervention Ranking

| Priority | Intervention | Target | Evidence | Expected Impact |
|----------|-------------|--------|----------|-----------------|
| 1 | Post-donation counseling (7-day follow-up) | First-time donors | Bagot 2016; Masser 2012 | High |
| 1 | SMS altruistic reminder | High churn risk | Yang 2020; Mohammed 2025 | High |
| 2 | Phone outreach | Inactive >12 months | Yang 2020 RCT | High |
| 3 | Camp notifications | `camp_ratio ≥ 0.60` | Srivastava 2025 (India) | Medium–High |
| 3 | Personalized invitation | Medium risk | Kauten 2021; Liu 2022 | Medium |
| 3 | Deferral-aware SMS (female donors) | Female + risk | Marwaha 2012; Sachdeva 2023 | Medium |
| 4 | Recognition / certificate | Frequent loyal donors | Malhotra 2026 | Medium |

## Model-Driven Workflow

1. Score all active donors weekly with `retained_180` model.
2. Merge RFM segment from `reports/rfm_segments.csv`.
3. Apply rules in `src/strategies/retention_engine.py`.
4. Export campaign list from `reports/donor_action_plan.csv`.
5. Track uplift by comparing predicted vs actual 180-day return.

## Literature Alignment

| Our Finding | Literature |
|-------------|------------|
| Camp/walk-in ratios among top SHAP drivers | India camp-channel segmentation (Mohammed 2025) |
| Recency and gap features important but not sole drivers | Yeh 2009; Kauten 2021 |
| Class imbalance requires recall-focused selection | Kiarie 2024 SR; Almutairi 2019 (SMOTE) |
| First-donation flag critical for rules | van Dongen 2015; Bagot 2016 |
