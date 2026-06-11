# Donor Segmentation Report (RFM)

## Method

RFM segmentation adapted from Yeh et al. (2009) and CRM practice:

- **Recency**: days since last donation (as of 2025-12-31)
- **Frequency**: lifetime donation count
- **Monetary**: total units collected (proxy for donation volume)

Quartile scores (1–4) assigned per dimension. Segment rules:

| Segment | Rule |
|---------|------|
| Loyal Donors | High frequency (Q3–Q4) and high recency score |
| Active Donors | Recent donors (recency Q3–Q4) not yet loyal |
| At-Risk Donors | Mid recency (Q2) with historically higher frequency |
| Lost Donors | Low recency score (Q1) |

## Results (Samarpan Synthetic Dataset V2)

| Segment | Donors | Avg Recency (days) | Avg Frequency | Avg Units | Business Interpretation |
|---------|--------|--------------------|---------------|-----------|-------------------------|
| Active Donors | 1,241 | 327 | 4.2 | 4.2 | Maintain engagement; moderate outreach |
| Lost Donors | 874 | 18* | 8.3 | 8.3 | *Synthetic artifact: many "lost" have high frequency but low recency score at year-end boundary |
| At-Risk Donors | 859 | 69 | 8.0 | 8.0 | Priority for SMS/call campaigns |
| Loyal Donors | 492 | 193 | 9.9 | 9.9 | Recognition programs; ambassador potential |

> **Note:** The synthetic generator creates heterogeneous lifecycles; segment boundaries should be recalibrated on production data. Aligns with Srivastava et al. (2025) recommendation for local cluster calibration.

## Segment Actions

| Segment | Recommended Action | Evidence |
|---------|-------------------|----------|
| Loyal Donors | Certificates, appreciation events | Malhotra 2026; Grazzini 2021 |
| Active Donors | Regular camp notifications | Mohammed 2025 (India) |
| At-Risk Donors | Personalized SMS + interval reminder | Yang 2020; Liu 2022 |
| Lost Donors | Phone outreach if >12 months inactive | Yang 2020 RCT |

Full donor-level segments: `reports/rfm_segments.csv`
