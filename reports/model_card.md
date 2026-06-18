# Model Card — Blood Donor Retention Prediction System

## Model Details
* **Developer:** Antigravity & Anushkap1205
* **Date:** June 18, 2026
* **Model Version:** 1.1 (Leakage-hardened, group-split, hyperparameter-tuned)
* **Model Types:** 
  * **180-Day Retention:** Logistic Regression (Tuned pipeline with StandardScaler and Categorical One-Hot Encoder)
  * **365-Day Churn:** CatBoost Classifier (Tuned pipeline with RandomizedSearchCV)

## Intended Use
* **Primary Objective:** Predict whether a blood donor will return to donate within 180 days after a given donation event (`retained_180`) or will lapse (no donation) for at least 365 days (`churn_365`).
* **Target Users:** Blood bank administrators, marketing campaign managers, and donor coordinators looking to schedule targeted communication campaigns.
* **Out-of-Scope:** Clinical decisions regarding donor health eligibility or immediate stock availability matching.

## Training Data & Validation Strategy
* **Dataset:** Samarpan Blood Bank Synthetic Dataset V2 (3,500 donors, 24,195 donation records).
* **Identity Leakage Prevention:** Standard row-based splits leak donor identity due to repeat donations (~6.9 average donations per donor). To prevent this, we split the data strictly by donor identity using **`StratifiedGroupKFold`** (5 folds) with `Donor_ID` as the group key. No donor's records appear in both the training and test splits.
* **Future Feature Leakage Prevention:** Removed the `Donation_Frequency_Label` (derived from the overall donor history) from the features list, restricting feature engineering strictly to historical observations prior to and including the anchor donation.
* **Hyperparameter Tuning:** Performed group-aware `RandomizedSearchCV` (5 iterations) for XGBoost, LightGBM, and CatBoost models.
* **Class Weighting:** Implemented class imbalance corrections across all models (e.g. `class_weight='balanced'` in LR/LGBM, `scale_pos_weight` dynamically computed in XGBoost, and `auto_class_weights='Balanced'` in CatBoost).

## Performance Leaderboards

### 180-Day Retention (Base Rate: 70.11%)
*Evaluation metrics computed on the group-held-out test set:*

| Model | ROC-AUC | PR-AUC | F1 | Recall | Precision | Accuracy | Brier Score |
|-------|---------|--------|----|--------|-----------|----------|-------------|
| **Logistic Regression** (Winner) | **0.6130** | **0.7622** | 0.7228 | 0.6807 | 0.7705 | 0.6339 | 0.2370 |
| Random Forest | 0.6122 | 0.7559 | 0.7703 | 0.7740 | 0.7665 | 0.6763 | 0.2246 |
| LightGBM | 0.6093 | 0.7500 | 0.7811 | 0.7954 | 0.7672 | 0.6873 | 0.2375 |
| CatBoost | 0.6080 | 0.7493 | 0.7811 | 0.7954 | 0.7672 | 0.6873 | 0.2374 |
| XGBoost | 0.6061 | 0.7463 | 0.7810 | 0.7944 | 0.7680 | 0.6876 | 0.2375 |

### 365-Day Churn (Base Rate: 10.22%)
*Evaluation metrics computed on the group-held-out test set:*

| Model | ROC-AUC | PR-AUC | F1 | Recall | Precision | Accuracy | Brier Score |
|-------|---------|--------|----|--------|-----------|----------|-------------|
| **CatBoost** (Winner) | **0.7007** | 0.1880 | **0.3226** | 0.6635 | **0.2131** | 0.7154 | 0.2217 |
| Logistic Regression | 0.6988 | **0.1926** | 0.3005 | **0.7013** | 0.1913 | 0.6666 | 0.2218 |
| XGBoost | 0.6945 | 0.1800 | 0.3223 | 0.6572 | 0.2135 | 0.7176 | 0.2196 |
| Random Forest | 0.6941 | 0.1736 | 0.2948 | 0.4937 | 0.2102 | 0.7588 | 0.1516 |
| LightGBM | 0.6865 | 0.1789 | 0.3223 | 0.6572 | 0.2135 | 0.7176 | 0.2207 |

## Probability Calibration
* **Brier Score:** Included for all models. The Brier score measures the mean squared difference between predicted probabilities and actual outcomes.
* **Calibration Plots:** Saved to `outputs/figures/[target_name]_[model_name]_curves.png`. These plots compare the fraction of positive instances against the mean predicted probability across 10 bins.
* **Calibration Utility:**
  * For **180-Day Retention**, the Random Forest model displays a slightly lower Brier score (0.2246) compared to the winner Logistic Regression (0.2370). Logistic Regression provides slightly better discrimination (ROC-AUC 0.6130 vs. 0.6122).
  * For **365-Day Churn**, probabilities are well-spread. The CatBoost model exhibits a balanced combination of Brier score (0.2217) and ROC-AUC (0.7007).

## Caveats & Limitations
* **Synthetic Data Limit:** The dataset is fully synthetic and simulated. Relationships between variables might not perfectly capture real human donation behaviors (e.g. scheduling constraints, health deferrals, or direct donation reminders).
* **Performance Gap vs. Literature:** Benchmark papers in the literature (e.g. Kauten et al. 2021) achieve MCC scores of 0.851 and AUCs above 0.80+. Our lower scores (0.61–0.70 AUC) highlight the limits of synthetic signals and the absence of rich clinical attributes (such as hemoglobin levels, historical deferrals, or response times to past marketing communications).
* **Recommendation:** Models should be refitted and validated against real, anonymized clinical/donation data before being deployed to a production setting.
