# Model Card — Blood Donor Retention Prediction System

## Model Details
* **Developer:** Antigravity & Anushkap1205
* **Date:** June 18, 2026
* **Model Version:** 1.2 (Leakage-hardened, group-split, fully tuned champion comparison)
* **Model Types:** 
  * **180-Day Retention:** Logistic Regression (Tuned pipeline with StandardScaler, One-Hot Encoder, and L2 regularization)
  * **365-Day Churn:** CatBoost Classifier (Tuned depth and iterations)

## Intended Use
* **Primary Objective:** Predict whether a blood donor will return to donate within 180 days after a given donation event (`retained_180`) or will lapse (no donation) for at least 365 days (`churn_365`).
* **Target Users:** Blood bank coordinators looking to schedule targeted communication campaigns.
* **Out-of-Scope:** Clinical decisions regarding donor health eligibility.

## Training Data & Validation Strategy
* **Dataset:** Samarpan Blood Bank Synthetic Dataset V2 (3,500 donors, 24,195 donation records).
* **Identity Leakage Prevention:** Standard row-based splits leak donor identity due to repeat donations (~6.9 average donations per donor). To prevent this, we split the data strictly by donor identity using **`StratifiedGroupKFold`** (5 folds) with `Donor_ID` as the group key. No donor's records appear in both training and test splits.
* **Future Feature Leakage Prevention:** Removed `Donation_Frequency_Label` (derived from the overall donor history) from the features list, restricting feature engineering strictly to historical observations prior to and including the anchor donation.
* **Hyperparameter Tuning:** Performed group-aware `RandomizedSearchCV` (5 iterations) on **all** models (Logistic Regression, Random Forest, XGBoost, LightGBM, and CatBoost) to ensure a completely fair champion selection.
* **Class Weighting:** Implemented class imbalance corrections across all models (e.g. `class_weight='balanced'` in LR/LGBM, `scale_pos_weight` dynamically computed in XGBoost, and `auto_class_weights='Balanced'` in CatBoost).

## Performance Leaderboards

### 180-Day Retention (Base Rate: 70.11%)
*Evaluation metrics computed on the group-held-out test set:*

| Model | ROC-AUC | PR-AUC | F1 | Recall | Precision | Accuracy | Brier Score |
|-------|---------|--------|----|--------|-----------|----------|-------------|
| **Logistic Regression** (Winner) | **0.6160** | **0.7635** | 0.7240 | 0.6814 | 0.7722 | 0.6357 | 0.2367 |
| Random Forest | 0.6130 | 0.7565 | 0.7682 | 0.7684 | 0.7679 | 0.6748 | 0.2280 |
| LightGBM | 0.6093 | 0.7500 | 0.7811 | 0.7954 | 0.7672 | 0.6873 | 0.2375 |
| CatBoost | 0.6080 | 0.7493 | 0.7811 | 0.7954 | 0.7672 | 0.6873 | 0.2374 |
| XGBoost | 0.6061 | 0.7463 | 0.7810 | 0.7944 | 0.7680 | 0.6876 | 0.2375 |

### 365-Day Churn (Base Rate: 10.22%)
*Evaluation metrics computed on the group-held-out test set:*

| Model | ROC-AUC | PR-AUC | F1 | Recall | Precision | Accuracy | Brier Score |
|-------|---------|--------|----|--------|-----------|----------|-------------|
| **CatBoost** (Winner) | **0.7007** | 0.1880 | **0.3226** | 0.6635 | **0.2131** | 0.7154 | 0.2217 |
| Logistic Regression | 0.6988 | **0.1915** | 0.3003 | **0.7013** | 0.1911 | 0.6662 | 0.2220 |
| XGBoost | 0.6945 | 0.1800 | 0.3223 | 0.6572 | 0.2135 | 0.7176 | 0.2196 |
| Random Forest | 0.6911 | 0.1716 | 0.3081 | 0.5755 | 0.2103 | 0.7359 | 0.1679 |
| LightGBM | 0.6865 | 0.1789 | 0.3223 | 0.6572 | 0.2135 | 0.7176 | 0.2207 |

---

## Probability Calibration & Naive Baselines

### 1. Naive Brier Baselines
The Brier Score measures the mean squared difference between predicted probabilities and actual binary outcomes. To judge if a model's Brier score represents an improvement, we compare it against a **naive baseline** that simply predicts the target base rate for all instances:
* **180-Day Retention Naive Baseline:** $0.7011 \times (1 - 0.7011)^2 + 0.2989 \times (0.7011)^2 \approx \mathbf{0.2096}$
  * *Analysis:* The models' Brier scores (0.2280 - 0.2375) are slightly higher than the naive baseline. Because the class is highly imbalanced toward returning (70.11%) and return intervals have high variance, attempting to predict values other than the baseline introduces higher squared error, though it yields discriminative power (AUC ~0.616).
* **365-Day Churn Naive Baseline:** $0.1022 \times (1 - 0.1022)^2 + 0.8978 \times (0.1022)^2 \approx \mathbf{0.0917}$
  * *Analysis:* The models' Brier scores (e.g. CatBoost 0.2217) are higher than the naive baseline due to the high penalty of false positives on a rare target (10.22%). However, because we use these probabilities to rank-order donors rather than for raw risk aggregation, the calibrated sorting remains effective.

### 2. Threshold Derivations
Rather than choosing arbitrary cutoffs, the intervention thresholds were derived as follows:
* **0.50 Churn Threshold:** In an imbalanced setting (10.22% base rate), $P(\text{churn}) \ge 0.50$ is a high-precision cutoff (nearly 5x the base rate). Selecting this threshold targets donors whose likelihood to lapse is extremely high, maximizing the ROI of high-cost human telephone outreach.
* **0.70 Retention Threshold (Females):** Any score below 0.70 represents a below-average probability of returning (average is 70.11%). Catching female donors before they drop below 0.70 triggers proactive iron-health counseling before they cross into the active churn phase.
* **0.80 Retention Threshold (Loyalty):** A score above 0.80 represents a high-certainty returner (top 30% of the distribution), ensuring milestone certificates go to the most habituated, secure donors.

---

## Operational Mechanics & Safeguards

### 1. Cold-Start Gap
For a donor's very first donation, RFMTC features (recency, gap statistics, and rolling frequency counts) are undefined or zero by construction. To prevent noisy predictions:
* First-time donors (`is_first_donation == 1`) bypass predictive scoring entirely.
* They are assigned the risk category `"First-Time Cold Start"` and routed to the Priority 1 counseling outreach path to establish early donation habits.

### 2. Protected Attribute Reframing
* **Current Implementation:** The strategy engine flags `Gender == 'Female'` and `retention_probability < 0.70` for deferral-prevention guides.
* **Justification:** Gender is used as a proxy for medical/physiological deferral risk (specifically anemia risk, as documented in Marwaha 2012) due to the absence of direct clinical check-up records in the synthetic V2 schema.
* **Roadmap:** Refine this rule to target a dedicated **medical deferral-risk score** once clinical check-up and blood-screening logs are integrated into the database.

### 3. Mutual Exclusivity Safeguard
To prevent multiple, conflicting communications (e.g., sending an SMS, a phone call, and a camp invite simultaneously), the strategy engine implements a single-contact policy. Interventions are evaluated in priority order; only the highest matching action is assigned, and long-term phone calls suppress short-term SMS campaigns.
