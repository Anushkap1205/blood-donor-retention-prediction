# Phase 1: Repository Scientific Audit Report

**System:** Samarpan Blood Bank Donor Retention Intelligence  
**Audit Date:** June 21, 2026  
**Auditor Role:** ML Engineering + Blood Bank Operations Research  

---

## Executive Summary

The repository implements a **research-guided retention prediction system** on a **synthetic dataset**. Core methodology (point-in-time features, donor-group splits) is **scientifically sound**. However, **moderate predictive performance** (retention ROC-AUC ~0.62, churn ROC-AUC ~0.70) reflects both synthetic data limitations and missing behavioral/operational variables present in real blood banks. The v2 production redesign addresses gaps in survival modeling, causal uplift, fairness auditing, and deployment architecture.

---

## 1. Code Review

### Architectural Issues

| Issue | Severity | Status |
|-------|----------|--------|
| Monolithic `src/` without clear inference/training separation | Medium | **Fixed** — `training/`, `inference/`, `recommendation_engine/` |
| No config-driven pipeline | Medium | **Fixed** — `config/pipeline.yaml` |
| Static HTML dashboard vs operational BI | Medium | **Fixed** — Streamlit dashboard |
| No experiment tracking / model registry | High | Partial — joblib artifacts + metrics JSON; MLflow recommended for production |
| Duplicate pipeline entry points | Low | `run_pipeline.py` (legacy) + `run_production_pipeline.py` (v2) |

### Data Leakage Risks

| Risk | Assessment | Mitigation |
|------|------------|------------|
| Donor_Status / Donation_Frequency labels in master table | **HIGH if used** | Correctly **excluded** from `get_feature_columns()` — these are post-hoc labels |
| Row-level train/test split | **HIGH** | **Fixed** — `StratifiedGroupKFold` on `Donor_ID` |
| Future donation information in features | **LOW** | Features computed only on history ≤ anchor date |
| Last_Donation_Date from master | **MEDIUM** | Not used directly; computed from register |
| Hyperparameter tuning on test fold | **MEDIUM** | Tuning uses CV on train fold only; test is single held-out group fold |

### Incorrect Preprocessing

- **Age imputation:** Missing ages not imputed in cleaning — acceptable with median imputation in pipeline.
- **StandardScaler on tree models:** Suboptimal but not invalid; tree models are scale-invariant post-encoding.
- **No explicit outlier capping** on `Units_Collected` — synthetic data shows no extreme outliers.

### Weak Feature Engineering (Pre-v2)

- Missing: seasonal effects, engagement/loyalty composites, communication proxies, deferral history.
- `donation_velocity` divides by tenure with floor — can inflate first-donation velocity artificially.
- No **time-to-event** target for survival analysis.

### Train/Test Split Methodology

- **Correct:** Donor-group stratified split prevents identity leakage.
- **Limitation:** Single fold used as test set (`next(sgkf.split(...))`) — results vary by fold; recommend reporting mean ± std across all 5 folds.

### Missing Validation Procedures (Pre-v2)

- No temporal validation (train on 2023–2024, test on 2025).
- No external validation cohort.
- No fairness auditing — **added in v2**.
- No survival model validation — **added in v2**.

### Overfitting Risks

- Random Forest/LightGBM/CatBoost show **higher train recall** with similar AUC — mild overfitting signal.
- Retention task: LR wins on AUC despite lower F1 — **prefer LR for ranking** (better calibrated for intervention prioritization).
- Churn task: rare class (10.22%) — high recall at cost of precision; **expected for screening use case**.

### Unrealistic Synthetic Data Assumptions

| Assumption | Reality Check |
|------------|-----------------|
| No deferral records | India: deferrals major dropout driver (Marwaha 2012; IJH 2021) |
| No communication logs | Cannot validate SMS/WhatsApp uplift on real response data |
| No satisfaction scores | Sachdeva 2023: feedback loops critical in India |
| No geographic distance | Shimla study: >10 km increases dropout |
| Camp_ID 44% null | Expected for walk-ins; camp assignment may be oversimplified |
| Uniform donor behavior | Real India: 73% dropout prevalence in some centres |

### Explainability Issues

- Global SHAP exists for tree models; LR champion for retention lacks native SHAP — permutation importance available.
- No **local donor narratives** — **added in v2** (`inference/explainer.py`).

### Scalability Concerns

- Observation dataset built with Python loop over 24K donations — O(n) per donor; acceptable at current scale.
- At 1M+ events: migrate to Spark/DuckDB feature store.
- Excel ingestion not production-viable — require PostgreSQL/API connector.

---

## 2. Dataset Review — Feature Decision Table

| Feature | Keep | Modify | Remove | Reason |
|---------|:----:|:------:|:------:|--------|
| Donor_ID | | | ✓ | Identifier only; used for grouping, not modeling |
| days_since_last_donation | ✓ | | | Strongest recency signal (Yeh 2009; Kauten 2021) |
| total_donations | ✓ | | | Core frequency feature |
| donations_last_3/6/12_months | ✓ | | | Rolling frequency; literature-validated |
| avg_gap_days, std_gap_days | ✓ | | | Inter-donation interval predictors (Liu 2022) |
| tenure_days / years_as_donor | ✓ | | | Donor maturity (RFMTC) |
| camp_ratio, walkin_ratio | ✓ | | | India-specific channel behavior (Srivastava 2025) |
| total_units_donated | ✓ | | | Monetary proxy in RFMTC |
| donation_velocity | | ✓ | | Modify: cap first-donation inflation |
| recent_activity_score | ✓ | | | Composite recency-frequency |
| is_first_donation | ✓ | | | Critical for cold-start routing (Bagot 2016) |
| donation_type_whole_blood_share | ✓ | | | Apheresis donors differ in return intervals |
| Age, Gender, Blood_Group | ✓ | | | Demographics; use with fairness audit |
| age_group | ✓ | | | Binned age for interpretability |
| Donor_Status (master) | | | ✓ | **Leakage** — post-hoc operational label |
| Donation_Frequency (master) | | | ✓ | **Leakage** — derived from full history |
| Last_Donation_Date (master) | | | ✓ | Redundant with register; leakage if used post-anchor |
| Camp_ID | ✓ | | | Add camp loyalty features in v3 |
| anchor_month/season | ✓ | | | **Added v2** — festival/seasonality (Marwaha 2015) |
| engagement_score | ✓ | | | **Added v2** — composite behavioral signal |
| loyalty_score | ✓ | | | **Added v2** — recognition program targeting |
| communication_response_rate | | ✓ | | **Proxy added** — replace with CRM logs |
| deferral_history | | | ✓ | **Not in data** — critical gap; add when available |
| satisfaction_score | | | ✓ | **Not in data** — add post-donation survey |
| travel_distance_km | | | ✓ | **Not in data** — add geocoding |
| reminder_response_rate | | ✓ | | **Proxy** — needs communication log |

---

## 3. Scientifically Invalid Methodology (Corrected)

1. ~~Using master-table status labels as features~~ — Already excluded; verified.
2. ~~Random row split~~ — Replaced with group split.
3. **Claiming WhatsApp effectiveness without RCT data** — v2 uplift uses literature priors with explicit confidence scores, not fabricated treatment outcomes.
4. **Gender as deferral proxy** — Documented as interim; requires clinical deferral records.

---

## 4. Recommendations

1. Deploy v2 production pipeline with enhanced features and fairness audit.
2. Integrate real Samarpan operational data before any campaign deployment.
3. Add temporal validation split when real multi-year data available.
4. Implement MLflow model registry for production versioning.
5. Run prospective RCT on SMS vs WhatsApp vs phone with holdout control.
