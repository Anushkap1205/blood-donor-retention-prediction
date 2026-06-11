# Blood Donor Retention — Literature Review (20+ Papers)

Research-guided summary for the Samarpan Blood Bank retention prediction system. Prioritized: Indian studies, developing-country evidence, systematic reviews, and highly cited foundational work.

## Summary Table

| # | Citation | Country | n | Objective | Key Retention Finding | ML / Stats | Dataset-Engineerable Features |
|---|----------|---------|---|-----------|----------------------|------------|------------------------------|
| 1 | Kauten et al. (2021), *Information Systems Frontiers* | USA | Large regional center | Predict donor return with ML | RF MCC=0.851; transaction history is primary signal | RF, LR, SVM | recency, frequency, volume, tenure |
| 2 | Yeh et al. (2009), *Expert Systems with Applications* / UCI-176 | Taiwan | 748 | RFMTC churn modeling | Recency + frequency dominate; 76% non-return class | Bernoulli sequence, LR, CART | R, F, M, T, churn label |
| 3 | Yang et al. (2020), *BMC Public Health* | China | 11,880 | Reactivate inactive donors (RCT) | Phone > SMS for long-inactive; male, older, prior donations predict return | ITT, ATT, logistic regression | months_since_last, gender, age, prior_count |
| 4 | Liu et al. (2022), *PLOS ONE* | China | 95,476 | Predict willingness to donate (COVID SMS) | Top predictors: interval, age, frequency; XGBoost AUC=0.809 | XGBoost, SVM, RF, LR | interval_days, age, frequency, blood_type |
| 5 | Charbonneau et al. (2016), *Transfusion Medicine Reviews* | Canada | 1,879 survey | Why donors lapse | Health reasons #1; gender-specific retention plans needed | Chi-square, descriptive | gender, age_group, lapse_flag |
| 6 | Srivastava et al. (2025), *Transfusion Clinique et Biologique* | India (South) | 1.5-yr prospective | Cluster donors by motivation/barriers | Segment-specific camp vs centre strategies | K-means / hierarchical clustering | camp_ratio, demographics, frequency |
| 7 | IJRSI comparative ML study (2025) | Generalizable | Donor history records | Benchmark LR vs RF vs SVM | RF accuracy ~92%, best non-linear fit | LR, RF, SVM | RFMTC-aligned history features |
| 8 | LMIC communication interventions SR (2025), *Vox Sanguinis* | Multi-LMIC | PRISMA review | Efficacy of outreach in LMICs | SMS/calls effective when culturally adapted | Systematic review | channel preference, recency |
| 9 | Marwaha & Sharma (2012), *Asian J Transfusion Sci* | India (North) | 37,896 | Adverse events and return | VVR 2.5%; adverse events linked to non-return | Descriptive, comparative | donation_type, adverse_event_flag |
| 10 | Ghana repeat-donor motivators study | Ghana | 100 | FT vs repeat motivators | Altruism universal; barriers differ by experience | Descriptive, comparative | is_first_donation, frequency |
| 11 | Veldhuizen et al. (2019) past-donation appeals | Netherlands | Field experiment | Reactivation messaging | Past-use appeals increase return | RCT, logistic models | prior_donation_count, recency |
| 12 | Marwaha (2015), *Asian J Transfusion Sci* | India (national) | NACO data | VBD status review | VBD 54%→79%; regional/seasonal variation | Policy review | seasonal_month, region |
| 13 | Zimbabwe time-series donation study | Zimbabwe | Multi-year | Forecast donation trends | Seasonal drops align with holidays | SARIMA, ETS | monthly_volume, seasonality |
| 14 | Sachdeva et al. (2023), India review | India | Review | Donor feedback → retention | Addressing suggestions improves satisfaction loop | Narrative review | satisfaction proxy unavailable in txn data |
| 15 | Callero & Piliavin (1999), identity theory | USA | Longitudinal | First donation → commitment | First experience shapes donor identity | Identity theory, regression | is_first_donation, tenure |
| 16 | van Dongen (2015), *Transfusion Medicine* | Netherlands | Review | Retention vs recruitment economics | Retention cheaper; lapsing peaks in first-time donors | Review, cost analysis | donation_count, is_first_donation |
| 17 | Bagot et al. (2016), *Transfusion Medicine Reviews* | Systematic review | Multi-study | First-time donor retention | Habit forms ~4th donation; FT strategies differ | Systematic review | total_donations, is_first_donation |
| 18 | Masser et al. (2012), *Transfusion* | Australia | TPB extension | Predict first-time donor retention | Extended TPB improves FT prediction | Logistic regression, TPB | demographics, intention proxies |
| 19 | Kiarie et al. (2024), systematic review | Kenya / global | Multi-study SR | ML for donor retention | RF, LR, SVM, ANN widely used; data quality critical | Systematic review | recency, frequency, demographics |
| 20 | Velasco Cardona et al. (2025), *Acta Haematologica Polonica* | Global SR | AI in blood banks | AI for retention & forecasting | Ensemble + time-series for retention | Systematic review | time-series, ensemble features |
| 21 | Ongo LightGBM study (2025), *IJCA* | Kenya | 5,000 | LightGBM retention prediction | LightGBM 98.3% accuracy on imbalanced data | LightGBM | 9 history features |
| 22 | Almutairi et al. (2019), *IJACSA* | Saudi Arabia | 2-year txn | Return donor classification | SMOTE needed; seasonal June/Sept drops | LR, RF, SVC + SMOTE | month, day-of-week, frequency |
| 23 | Mohammed et al. (2025), *Transfusion Clinique et Biologique* | India (Manipal) | Single-centre | Donor stratification | SMS preferred; cluster-based messaging | Cluster analysis | camp_ratio, demographics |
| 24 | Malhotra et al. (2026), *Asian J Transfusion Sci* | India (North) | Single-centre | Incentive attitudes | Recognition > monetary incentives locally | Survey | recognition_eligible (frequency proxy) |
| 25 | Pune motivators study (2018), *Indian J Community Med* | India | 181 | Motivation PCA | Altruism #1; repeat vs FT similar drivers | PCA | age, gender, repeat_flag |
| 26 | Mass counseling India (2014), PMC | India | Community | Counseling → repeat donation | Self-identity via counseling improves retention | Quasi-experimental | is_first_donation |
| 27 | Temporary deferral review (2021), *Indian J Hematol* | India | Review | TD impact on return | FT and young donors less likely to return after deferral | Narrative review | deferral_count (not in dataset) |
| 28 | Ferguson et al. (2007), *Transfusion* | Theory | Review | Integrate theory into recruitment | TPB + identity improve intervention design | Theoretical framework | behavioral proxies from history |

## Practical Ranking for This Project

| Rank | Paper | Why |
|------|-------|-----|
| 1 | Yeh et al. (2009) / UCI RFMTC | Direct blueprint for recency-frequency-volume-tenure features |
| 2 | Kauten et al. (2021) | Gold-standard ML retention pipeline on transaction logs |
| 3 | Liu et al. (2022) | Validates XGBoost + interval/frequency/age triad |
| 4 | Srivastava / Mohammed (India) | India-specific segmentation and camp-channel effects |
| 5 | Yang et al. (2020) | Evidence-based intervention ranking (SMS vs phone) |
| 6 | van Dongen / Bagot | First-donation and habit-formation business rules |
| 7 | Kiarie / Velasco systematic reviews | Model selection and evaluation best practices |

Full HTML profiles: `reports/blood_donor_retention_research.html`
