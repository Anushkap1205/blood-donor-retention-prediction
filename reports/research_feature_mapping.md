# Research-to-Feature Mapping

Only features derivable from **Donor_Master** and **Donation_Register** are retained.

| Research Finding | Source | Dataset Variable | Engineered Feature | Expected Impact |
|------------------|--------|------------------|--------------------|-----------------|
| Recency is strongest return predictor | Yeh 2009; Kauten 2021; Liu 2022 | Donation_Date | `days_since_last_donation` | Negative on retention |
| Donation frequency predicts loyalty | Yeh 2009; Bagot 2016 | Donor_ID, Donation_Date | `total_donations`, `donations_last_*` | Positive |
| Habit forms after ~4 donations | van Dongen 2015; Bagot 2016 | Donation history | `total_donations`, `is_first_donation` | Positive after threshold |
| Inter-donation interval predicts return | Liu 2022; Almutairi 2019 | Donation_Date gaps | `avg_gap_days`, `std_gap_days`, `min_gap_days`, `max_gap_days` | Non-linear |
| Tenure / time since first donation | Yeh 2009 RFMTC | Registration_Date, Donation_Date | `tenure_days`, `years_as_donor`, `days_since_registration` | Positive (experienced donors) |
| Volume donated (RFM Monetary) | Yeh 2009; Kauten 2021 | Units_Collected | `total_units_donated`, `average_units_per_donation` | Positive |
| Camp vs walk-in channel preference | Srivastava 2025; Mohammed 2025 (India) | Collection_Source | `camp_ratio`, `walkin_ratio`, `camp_donation_count` | Segment-specific |
| Age affects return probability | Liu 2022; Yang 2020 | Age | `Age`, `age_group` | Non-linear |
| Gender affects deferral/return | Marwaha 2012; Yang 2020 | Gender | `Gender` (one-hot) | Segment-specific |
| Blood group supply-demand effects | Kauten 2021 | Blood_Group | `Blood_Group` (one-hot) | Weak–moderate |
| Recent activity indicates engagement | Kauten 2021 | Rolling donations | `recent_activity_score`, `donation_velocity` | Positive |
| First-time donors highest lapse risk | van Dongen 2015 | Donation count at anchor | `is_first_donation` | Negative on retention |
| Whole blood vs apheresis behavior differs | Charbonneau 2016 | Donation_Type | `donation_type_whole_blood_share` | Weak |
| Seasonal donation variation | Almutairi 2019; Marwaha 2015 | Donation_Date | *(excluded from anchor model; used in EDA)* | Planning only |
| SMS/phone outreach by inactivity duration | Yang 2020 | Derived risk score | `retention_probability` → intervention rules | Operational |

## Features Excluded (Not in Transaction Data)

| Construct | Source | Reason |
|-----------|--------|--------|
| Motivation / altruism score | Pune 2018; Mohammed 2025 | Survey-only |
| Donor satisfaction | Sachdeva 2023 | No feedback table |
| Adverse reaction at donation | Marwaha 2012 | Not recorded in register |
| Deferral history | Indian deferral review 2021 | Not in synthetic dataset |
| SMS response history | Liu 2022 | No campaign log |
