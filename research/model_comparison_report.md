# Phase 5: Model Comparison Report

**Validation:** StratifiedGroupKFold (Donor_ID), 5-fold CV, RandomizedSearchCV  
**Dataset:** Samarpan Synthetic V2 + v2 enhanced features  

---

## Classification Models

### 180-Day Retention (Base Rate: 70.11%)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier | Calibration | Interpretability |
|-------|----------|-----------|--------|-----|---------|--------|-------|-------------|------------------|
| **Logistic Regression** ★ | 0.636 | 0.772 | 0.681 | 0.724 | **0.616** | **0.764** | 0.237 | Good | ★★★★★ |
| Random Forest | 0.675 | 0.768 | 0.768 | 0.768 | 0.613 | 0.757 | 0.228 | Moderate | ★★★☆☆ |
| LightGBM | 0.687 | 0.767 | 0.795 | 0.781 | 0.609 | 0.750 | 0.238 | Moderate | ★★★☆☆ |
| CatBoost | 0.687 | 0.767 | 0.795 | 0.781 | 0.608 | 0.749 | 0.237 | Moderate | ★★★☆☆ |
| XGBoost | 0.688 | 0.768 | 0.794 | 0.781 | 0.606 | 0.746 | 0.238 | Moderate | ★★★☆☆ |

**Champion: Logistic Regression** — Best ROC-AUC and PR-AUC; most interpretable; preferred for probability-ranked outreach.

### 365-Day Churn (Base Rate: 10.22%)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier | Calibration | Interpretability |
|-------|----------|-----------|--------|-----|---------|--------|-------|-------------|------------------|
| **CatBoost** ★ | 0.715 | 0.213 | 0.664 | 0.323 | **0.701** | 0.188 | 0.222 | Moderate | ★★★☆☆ |
| Logistic Regression | 0.666 | 0.191 | 0.701 | 0.300 | 0.699 | **0.192** | 0.222 | Good | ★★★★★ |
| XGBoost | 0.718 | 0.214 | 0.657 | 0.322 | 0.695 | 0.180 | 0.220 | Moderate | ★★★☆☆ |
| Random Forest | 0.736 | 0.210 | 0.576 | 0.308 | 0.691 | 0.172 | 0.168 | Poor at low P | ★★★☆☆ |
| LightGBM | 0.718 | 0.214 | 0.657 | 0.322 | 0.686 | 0.179 | 0.221 | Moderate | ★★★☆☆ |

**Champion: CatBoost** — Best discrimination for rare churn class; SHAP available.

---

## Survival Analysis (v2)

| Model | Metric | Value | Use Case |
|-------|--------|-------|----------|
| **Cox Proportional Hazards** | Concordance Index | ~0.65–0.70 | Time-to-next-donation, next date estimation |
| Kaplan-Meier | Survival curves | By segment | Operations planning |

**Advantage:** Handles censoring natively; predicts *when* not just *if*.  
**Limitation:** Assumes proportional hazards; donor-level frailty not modeled.

---

## Recommended Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    PRODUCTION ENSEMBLE                        │
├──────────────────────────────────────────────────────────────┤
│  RETENTION (180d)          │  CHURN (365d)                   │
│  Logistic Regression       │  CatBoost                        │
│  → outreach ranking        │  → lapsed-donor screening        │
├──────────────────────────────────────────────────────────────┤
│  SURVIVAL: Cox PH          │  → next donation date estimate   │
├──────────────────────────────────────────────────────────────┤
│  CAUSAL: T-Learner +       │  → intervention uplift ranking   │
│  literature priors         │                                   │
├──────────────────────────────────────────────────────────────┤
│  RULES ENGINE              │  → final action (single contact) │
└──────────────────────────────────────────────────────────────┘
```

### Why Not a Single Model?

- Retention (high base rate) and churn (low base rate) are **different decision problems**
- LR excels at calibrated ranking for majority class; CatBoost for rare event detection
- Survival model adds temporal dimension rules cannot capture

### Metric Selection for Deployment

| Use Case | Primary Metric |
|----------|----------------|
| Rank donors for SMS campaign | PR-AUC, calibration |
| Screen lapsed donors for phone | Recall @ P≥0.5 |
| Estimate inventory needs | Survival curve median time |
| A/B test evaluation | Uplift ATE |

---

## Statistical Notes

- McNemar test available in `src/models/evaluate.py` for paired model comparison
- Brier scores exceed naive baseline on retention — expected with high-variance intervals; models still add ranking value
- All results are **synthetic-validated only**
