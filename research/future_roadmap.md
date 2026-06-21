# Future Improvements Roadmap

## Q3 2026 — Data Integration

| Priority | Item | Impact |
|----------|------|--------|
| P0 | Replace synthetic data with Samarpan production DB | Enables real deployment |
| P0 | Integrate deferral history from screening logs | High — top India dropout driver |
| P1 | SMS/WhatsApp gateway response tracking | Enables real uplift modeling |
| P1 | Post-donation satisfaction survey (NPS) | Sachdeva 2023 evidence |

## Q4 2026 — Model Maturity

| Priority | Item | Impact |
|----------|------|--------|
| P0 | Prospective RCT: SMS vs WhatsApp vs phone | Validates causal claims |
| P1 | MLflow model registry + automated retraining | Production MLOps |
| P1 | Temporal validation (train past, test future) | Realistic performance estimate |
| P2 | Deep survival (DeepHit, random survival forests) | Better time-to-event |
| P2 | Donor-level mixed effects frailty model | Handles repeat measures |

## Q1 2027 — Scale & Intelligence

| Priority | Item | Impact |
|----------|------|--------|
| P1 | Geospatial distance features (urban/rural) | Shimla study evidence |
| P1 | Inventory-linked demand signaling | Veldhuizen 2019 appeals |
| P2 | Multi-blood-bank federated learning | Privacy-preserving scale |
| P2 | Real-time streaming feature store (Kafka) | Sub-daily scoring |
| P3 | LLM-generated personalized messages (human-reviewed) | Personalization at scale |

## Q2 2027 — Governance

| Priority | Item | Impact |
|----------|------|--------|
| P0 | Replace gender deferral proxy with clinical score | Fairness + accuracy |
| P1 | External audit of fairness metrics | Regulatory compliance |
| P1 | ISO 15189 alignment for blood bank AI tools | Quality management |
| P2 | Explainability dashboard for coordinators | Trust + adoption |

## Known Limitations (Current v2)

1. Uplift effects use **literature priors**, not observed treatment outcomes
2. Communication proxies are **synthetic stand-ins**
3. Single held-out fold — should report cross-fold mean ± std
4. No integration with hospital demand forecasting
5. Festival season flag is heuristic, not calendar-specific

## Success Metrics (Real Deployment)

| KPI | Target | Measurement |
|-----|--------|-------------|
| 180-day retention rate | +5% absolute | A/B vs control |
| Cost per retained donor | −20% | Campaign spend / returns |
| First-time donor return | +10% | FT cohort tracking |
| Female deferral return rate | +8% | Deferral-aware program |
| Model ROC-AUC (retention) | > 0.75 | Real data validation |
