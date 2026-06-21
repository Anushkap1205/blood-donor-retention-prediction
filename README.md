# Blood Donor Retention Prediction System

> [!WARNING]
> **Synthetic Dataset Caveat:**
> The Samarpan Blood Bank dataset is simulated/synthetic. All models, segmentation algorithms, and intervention rules have been validated only against this synthetic dataset. They must be validated against real clinical/operational records before any real-world clinical or campaign deployment.

Research-guided machine learning system for predicting blood donor return within **180 days** and churn within **365 days**, built on the Samarpan Blood Bank synthetic dataset.


## Features

- Literature-backed feature engineering (RFMTC, India-specific camp/channel effects)
- Point-in-time observation dataset (no leakage)
- Models: Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost
- Evaluation: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, 5-fold CV
- SHAP + permutation importance
- RFM donor segmentation (Loyal / Active / At-Risk / Lost)
- Retention strategy engine with evidence-ranked interventions

## Project Structure (v2 Production)

```
├── config/pipeline.yaml           # Central configuration
├── data/                          # Samarpan Excel dataset
├── research/                      # Audit reports, architecture, deployment guide
├── feature_store/                 # Enhanced RFMTC+ feature engineering
├── training/                      # Survival, uplift, fairness modules
├── inference/                     # Scoring + local SHAP explanations
├── recommendation_engine/         # Rules + causal intervention engine
├── dashboard/streamlit_app.py     # Executive / Donor / Operations views
├── src/                           # Core library (data, models, EDA)
├── scripts/
│   ├── run_pipeline.py            # Legacy end-to-end pipeline
│   └── run_production_pipeline.py # v2 production orchestrator
├── tests/                         # Unit tests
└── outputs/                       # Models, metrics, figures
```

## Quick Start

```bash
# macOS: boosting libraries may need OpenMP
brew install libomp

pip install -r requirements.txt

# v2 production pipeline (recommended)
python scripts/run_production_pipeline.py

# Streamlit dashboard
streamlit run dashboard/streamlit_app.py

# Unit tests
pytest tests/ -v
```

## Dataset

**File:** `data/Samarpan_BloodBank_SyntheticDataset_V2.xlsx`

| Sheet | Rows | Use |
|-------|------|-----|
| Donor_Master | 3,500 | Demographics & status |
| Donation_Register | 24,195 | Transaction history |

## Targets

| Target | Definition |
|--------|------------|
| `retained_180` | 1 if next donation within 180 days after anchor donation |
| `churn_365` | 1 if no donation within 365 days after anchor donation |

## Research & Audit Reports (v2)

| Report | Description |
|--------|-------------|
| `research/repository_audit_report.md` | Phase 1 scientific code & data audit |
| `research/retention_drivers_ranked.md` | Phase 2 literature-backed driver ranking |
| `research/india_retention_strategy_framework.md` | Phase 3 India-specific interventions |
| `research/feature_engineering_report.md` | Phase 4 feature formulas & rationale |
| `research/model_comparison_report.md` | Phase 5 model selection architecture |
| `research/architecture.md` | Phase 10 system architecture diagram |
| `research/deployment_guide.md` | Production deployment instructions |
| `research/future_roadmap.md` | Phase 12 improvement roadmap |
| `reports/research_review.md` | 28-paper literature summary |
| `reports/research_feature_mapping.md` | Evidence → feature mapping |
| `reports/data_dictionary.md` | Schema documentation |
| `reports/segmentation_report.md` | RFM segment analysis |
| `reports/retention_strategy_recommendations.md` | Intervention playbook |
| `reports/final_research_report.md` | Full research-style write-up |
| `reports/donor_action_plan.csv` | Scored donors + recommended actions |

## Model Selection & Calibration (Hardened Run)

* **Validation Strategy:** Evaluated using **5-fold `StratifiedGroupKFold`** grouped on `Donor_ID` to prevent donor identity leakage, alongside randomized hyperparameter search.
* **Probability Calibration:** Tracked Brier scores and generated calibration curves to ensure predicted risk probabilities are reliable for ranking intervention targets.

| Problem (Base Rate) | Champion Model | test ROC-AUC | test PR-AUC | Brier Score | Path |
|---------|----------------|--------------|-------------|-------------|------|
| 180-day retention (70.11%) | **Logistic Regression** | **0.6160** | **0.7635** | 0.2367 | `outputs/models/retained_180_logistic_regression.joblib` |
| 365-day churn (10.22%) | **CatBoost** | **0.7007** | **0.1880** | 0.2217 | `outputs/models/churn_365_catboost.joblib` |

For full splits, parameter search grids, metrics, and calibration curves, see [model_card.md](file:///Users/anushkapatil/Projects/blood-donor-retention-prediction/reports/model_card.md).


## References

Key papers: Yeh et al. (2009) RFMTC; Kauten et al. (2021); Liu et al. (2022); Yang et al. (2020); Mohammed et al. (2025, India); van Dongen (2015); Bagot et al. (2016).

Full bibliography: `reports/research_review.md` and `reports/blood_donor_retention_research.html`.

## License

Academic / educational use. Dataset is synthetic for the Samarpan Blood Bank case study.
