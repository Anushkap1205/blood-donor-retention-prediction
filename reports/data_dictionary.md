# Data Dictionary — Samarpan Blood Bank Synthetic Dataset V2

## Donor_Master

| Column | Type | Description | Keys / Notes |
|--------|------|-------------|--------------|
| Donor_ID | string | Unique donor identifier | **Primary key** (e.g. DON00001) |
| Gender | categorical | Male / Female | Demographic feature |
| Age | integer | Donor age in years | Demographic feature |
| Blood_Group | categorical | ABO/Rh group | Demographic feature |
| Registration_Date | date | First registration | Tenure anchor |
| Donation_Frequency | categorical | Rare / Occasional / Regular label | Summary label from generator |
| Last_Donation_Date | date | Most recent donation | Cross-check with register |
| Donor_Status | categorical | Active / Inactive / Lapsed | Operational status |

**Rows:** 3,500 | **Duplicates:** 0 | **Missing:** none

## Donation_Register

| Column | Type | Description | Keys / Notes |
|--------|------|-------------|--------------|
| Donation_ID | string | Unique donation event | **Primary key** |
| Donation_Date | date | Date of collection | Time-series anchor |
| Donor_ID | string | Donor reference | **Foreign key → Donor_Master** |
| Blood_Group | categorical | Group at donation | May differ if updated |
| Units_Collected | numeric | Units per event | RFM monetary proxy |
| Donation_Type | categorical | Whole Blood / Apheresis | Behavioral |
| Collection_Source | categorical | Camp / Walk-in | Channel feature |
| Camp_ID | string | Camp identifier | Null for walk-ins (10,759 nulls) |

**Rows:** 24,195 | **Date range:** 2023-01-02 to 2025-12-31 | **Duplicates:** 0

## Supporting Tables (not used in retention model)

- **Patient_Master** (140 rows): transfusion patients
- **Hospital_Master** (19 rows): requesting hospitals
- **Blood_Issue_Register** (12,369 rows): issue transactions
- **Monthly_Inventory** (288 rows): stock by month and blood group

## Data Quality Summary

| Check | Result |
|-------|--------|
| Orphan donations (no donor master) | 0 |
| Donors without any donation | 34 |
| Duplicate Donation_ID | 0 |
| Camp_ID missing (walk-ins) | Expected |

Machine-readable dictionary: `reports/data_dictionary.json`
