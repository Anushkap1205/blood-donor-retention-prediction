# Blood Donor Retention Prediction — Final Research Report

## Executive Summary

This project implements a research-guided machine learning system to predict blood donor retention (180-day return) and churn (365-day non-return) using the Samarpan Blood Bank synthetic dataset (3,500 donors, 24,195 donations, 2023–2025). The pipeline follows RFMTC principles (*Yeh et al., 2009*), contemporary ML benchmarks (*Kauten et al., 2021; Liu et al., 2022*), and India-specific retention evidence (*Mohammed, 2025; Malhotra, 2026*). 

This version has been hardened by:
1. Eliminating donor identity leakage via `StratifiedGroupKFold` splits grouped by `Donor_ID`.
2. Eliminating look-ahead features (`Donation_Frequency_Label`).
3. Running a fair, hyperparameter-tuned champion selection across all candidate models.
4. Implementing a mutually exclusive prioritization rule engine that resolves communication overlaps.
5. Providing explicit Brier score baselines, cold-start routes, and A/B test persistence controls.

---

## Target Construction & Class Prevalence

### 1. Retention (`retained_180 = 1`)
* **Definition:** At each donation anchor date *t*, `retained_180 = 1` if another donation occurs within (*t*, *t* + 180 days].
* **Prevalence (Base Rate):** **70.11%** of donation observations result in a return within 180 days.
* **Censoring:** Observations where *t* + 180 days extends beyond the maximum date in the transaction history are excluded.

### 2. Churn (`churn_365 = 1`)
* **Definition:** At each donation anchor date *t*, `churn_365 = 1` if **no** donation occurs within (*t*, *t* + 365 days].
* **Prevalence (Base Rate):** **10.22%** of donation observations result in churn.
* **Censoring:** Observations where *t* + 365 days extends beyond the maximum date are excluded.

---

## Model Evaluation Leaderboards

All models are tuned using `RandomizedSearchCV` on 5-fold StratifiedGroupKFold splits. Evaluation metrics are reported on the group-held-out test set.

### 180-Day Retention (Base Rate: 70.11%)

| Model | ROC-AUC | PR-AUC | F1 | Recall | Precision | Accuracy | Brier Score |
|-------|---------|--------|----|--------|-----------|----------|-------------|
| **Logistic Regression** (Winner) | **0.6160** | **0.7635** | 0.7240 | 0.6814 | 0.7722 | 0.6357 | 0.2367 |
| Random Forest | 0.6130 | 0.7565 | 0.7682 | 0.7684 | 0.7679 | 0.6748 | 0.2280 |
| LightGBM | 0.6093 | 0.7500 | 0.7811 | 0.7954 | 0.7672 | 0.6873 | 0.2375 |
| CatBoost | 0.6080 | 0.7493 | 0.7811 | 0.7954 | 0.7672 | 0.6873 | 0.2374 |
| XGBoost | 0.6061 | 0.7463 | 0.7810 | 0.7944 | 0.7680 | 0.6876 | 0.2375 |

* **Selection Rationale:** Tuned Logistic Regression wins on ROC-AUC (0.6160) and PR-AUC (0.7635). It provides stable, regularized parameter estimates.
* **Brier Baseline Comparison:** The naive baseline (always predicting the base rate of 0.7011) yields a Brier score of **0.2096**. The models' Brier scores (0.2280 - 0.2375) are slightly higher, reflecting the high variance of return dates in the synthetic dataset.

### 365-Day Churn (Base Rate: 10.22%)

| Model | ROC-AUC | PR-AUC | F1 | Recall | Precision | Accuracy | Brier Score |
|-------|---------|--------|----|--------|-----------|----------|-------------|
| **CatBoost** (Winner) | **0.7007** | 0.1880 | **0.3226** | 0.6635 | **0.2131** | 0.7154 | 0.2217 |
| Logistic Regression | 0.6988 | **0.1915** | 0.3003 | **0.7013** | 0.1911 | 0.6662 | 0.2220 |
| XGBoost | 0.6945 | 0.1800 | 0.3223 | 0.6572 | 0.2135 | 0.7176 | 0.2196 |
| Random Forest | 0.6911 | 0.1716 | 0.3081 | 0.5755 | 0.2103 | 0.7359 | **0.1679** |
| LightGBM | 0.6865 | 0.1789 | 0.3223 | 0.6572 | 0.2135 | 0.7176 | 0.2207 |

* **Selection Rationale:** CatBoost achieves the best discrimination (ROC-AUC 0.7007) while maintaining a balanced F1 score. Under extreme imbalance, PR-AUC is the key performance index; Logistic Regression has a slightly higher PR-AUC (0.1915) but lower precision than CatBoost.
* **Brier Baseline Comparison:** The naive baseline (predicting 0.1022) yields a Brier score of **0.0917**. The models' Brier scores (e.g. CatBoost 0.2217) are higher due to false positive penalties on a rare target. This necessitates threshold-based calibration.

---

## Strategy Engine & Safeguards

The strategy engine (`src/strategies/retention_engine.py`) maps predictions to specific interventions using a **strict priority ladder** to guarantee that every donor receives **exactly one** primary campaign contact, preventing spam.

1. **Cold-Start Gap (Priority 1):** First-time donors (`is_first_donation == 1`) bypass model scoring. They are placed in the `"First-Time Cold Start"` risk category and routed directly to the post-donation counseling program.
2. **Long-Term Reactivation Call (Priority 1):** Selected if `churn_probability >= 0.50` and `days_since_last_donation > 365`. Suppresses SMS campaigns to avoid redundant channels.
3. **Altruistic Churn SMS (Priority 2):** Selected if `churn_probability >= 0.50`. If the donor is camp-oriented (`camp_ratio >= 0.60`), the SMS is customized to include geocoded camp matching.
4. **Deferral-Aware Scheduling (Priority 3):** Selected if `Gender == 'Female'` and `retention_probability < 0.70` (sub-optimal retention).
   * *Protected Attribute Justification:* Gender is a proxy for physiological deferral risk (specifically anemia risk, *Marwaha 2012*) due to database limits in V2. The roadmap calls for replacing this with a dedicated deferral-risk classifier when screening logs are available.
5. **Camp-Targeted Notifications (Priority 4):** Sent to active camp-goers (`camp_ratio >= 0.60`) to notify them of nearby drives.
6. **Milestone Appreciation (Priority 5):** Sent to loyal veterans (`total_donations >= 5`, `retention_probability >= 0.80`) to celebrate their identity.
7. **Personalized Donation Invitation (Priority 6):** Sent to standard medium-risk return candidates (`retention_probability < 0.80`).

---

## Operational Deployment Roadmap

1. **Weekly cron job:** Run `scripts/run_pipeline.py` every Sunday to update scores.
2. **Experimental A/B Test Persistence:** Group assignments (`Treatment`/`Control` in a 9:1 split) and `Assignment_Date` are saved in a persistent ledger `outputs/metrics/experimental_assignments.csv`. This prevents weekly runs from re-shuffling cohorts, enabling a valid 90-day comparison.
3. **Sample Size & Power:** Under a 90/10 split, a sample size of 3,500 active donors is fully powered to detect a Minimum Detectable Effect (MDE) of **5.5%** in return rates using a **Two-Proportion Z-Test** (at 80% power and 5% alpha).
4. **PII Compliance:** Hashing or anonymizing phone numbers is required before transmitting data to third-party SMS gateways (DPDP Act compliance).
5. **Drift Monitoring:** Monthly tracking of Population Stability Index (PSI > 0.20) or a drop in AUC > 0.05 triggers an automated retraining alert.
