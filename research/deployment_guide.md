# Production Deployment Guide

## Prerequisites

```bash
# macOS OpenMP for boosting libraries
brew install libomp

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Full Pipeline

```bash
# v2 production pipeline (enhanced features, survival, uplift, fairness)
python scripts/run_production_pipeline.py

# Legacy pipeline (still functional)
python scripts/run_pipeline.py
```

## Launch Dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

## Run Tests

```bash
pytest tests/ -v
```

## Deployment Options

### Option A: Batch (Recommended for Phase 1)

1. Schedule `run_production_pipeline.py` via cron/Airflow (weekly)
2. Output `reports/donor_action_plan.csv` to blood bank CRM
3. Streamlit dashboard on internal server

### Option B: API Service

Wrap `inference/scorer.py` in FastAPI:

```python
from fastapi import FastAPI
from inference.scorer import DonorScorer

app = FastAPI()
scorer = DonorScorer("outputs/models")
scorer.load()

@app.post("/score")
def score(donor_features: dict):
    # Transform and return probabilities
    ...
```

### Option C: Cloud (AWS/GCP)

| Component | Service |
|-----------|---------|
| Data store | S3 / GCS + PostgreSQL |
| Training | SageMaker / Vertex AI |
| Inference | Lambda / Cloud Run |
| Dashboard | Streamlit Cloud / internal EC2 |

## Pre-Production Checklist

- [ ] Replace synthetic Excel with real PostgreSQL connector
- [ ] Validate models on ≥6 months real donor data
- [ ] Run prospective A/B test (90/10 treatment/control)
- [ ] Integrate deferral and communication logs
- [ ] Legal review: DPDP consent for outreach channels
- [ ] Set up PSI monitoring dashboard
- [ ] Register models in MLflow

## Rollback Procedure

1. Retain previous model artifacts with date suffix
2. Update `config/pipeline.yaml` champion model names
3. Re-run inference only (skip training) against frozen feature store snapshot

## Expected Outputs After Pipeline

| File | Description |
|------|-------------|
| `feature_store/artifacts/observation_dataset.parquet` | Full feature matrix |
| `reports/donor_action_plan.csv` | Scored donors + interventions |
| `reports/causal_recommendations.csv` | Uplift-based actions |
| `reports/donor_explanations.json` | Local SHAP narratives |
| `outputs/metrics/*_fairness.json` | Bias audit results |
| `outputs/metrics/survival_cox.json` | Survival model metrics |
