# Phase 4: Feature Engineering Report

---

## Synthetic Data Fidelity Assessment

The Samarpan V2 dataset captures **core RFMTC dynamics** (recency, frequency, volume, tenure, camp channel) consistent with Yeh 2009 and Kauten 2021. It **does not capture**:

- Deferral cascades (major India dropout driver)
- Communication response logs
- Satisfaction / waiting time
- Geographic distance

**Conclusion:** Features below improve predictive power **within synthetic constraints**; real-world lift requires operational data integration.

---

## Engineered Features — Formulas & Rationale

| Feature | Formula | Predictive Rationale |
|---------|---------|---------------------|
| **donation_consistency** | `1 / (1 + std_gap / avg_gap)` | Regular donors return more (Liu 2022 interval predictor) |
| **preferred_donation_interval_days** | `mean(diff(donation_dates))` | Personal cadence for reminder timing |
| **communication_response_rate_proxy** | `0.6 × recent_activity + 0.4 × camp_ratio` | Proxy until CRM data; camp donors respond to outreach |
| **campaign_response_history** | `camp_donations / total_donations` | India camp-oriented segments (Srivastava 2025) |
| **appointment_attendance_rate_proxy** | `0.5 × walkin_ratio + 0.5 × consistency` | Walk-in implies self-initiation; inverse no-show proxy |
| **engagement_score** | `0.35×recent_activity + 0.25×velocity + 0.20×consistency + 0.20×camp_ratio` | Composite behavioral engagement |
| **loyalty_score** | `(total/10) × (1 / (1 + recency/180))` | Recognition program eligibility (Malhotra 2026) |
| **donor_lifetime_value** | `total_units × (1 + 0.1 × total_donations)` | Prioritize high-value donor retention |
| **is_regular_giver** | `1 if 56 ≤ avg_gap ≤ 120 and total ≥ 3` | NACO 90-day minimum gap compliance |
| **gap_trend_days** | `days_since_last − avg_gap` | Positive = overdue vs personal pattern |
| **anchor_sin/cos_month** | Cyclical month encoding | Seasonal donation patterns (Marwaha 2015) |
| **is_festival_season** | `month ∈ {3,4,10,11,12}` | Indian festival camp peaks |

---

## Features Requiring Real Data (Roadmap)

| Feature | Required Source | Expected Impact |
|---------|-----------------|-----------------|
| deferral_count | Clinical screening logs | High |
| reminder_response_rate | SMS/WhatsApp gateway | High |
| satisfaction_score | Post-donation survey | Medium-High |
| travel_distance_km | Donor address geocoding | Medium |
| adverse_event_flag | Donation reaction register | Medium |
| referral_count | Referral program CRM | Medium |

---

## Leakage Prevention Checklist

- [x] All features computed at anchor_date using history ≤ anchor
- [x] Master-table post-hoc labels excluded
- [x] Right-censored labels excluded (NaN when window exceeds data horizon)
- [x] Group split by Donor_ID
