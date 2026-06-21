# Phase 3: India-Specific Retention Strategy Framework

---

## Context: Indian Blood Donation Ecosystem

| Factor | Evidence | Implication |
|--------|----------|-------------|
| Voluntary donor dropout 70–73% in some centres | Shimla blood bank study | Retention > recruitment ROI |
| VBD rose 54%→79% (2007–2022) | Marwaha 2015; NACO | Growing pool but high first-time attrition |
| South Asia: 5–10% repeat donors historically | Bharucha 2005 | Massive retention opportunity |
| WhatsApp ~95% preferred IMA among Indian blood banks | PubMed 2024 survey | Channel exists; retention efficacy unproven |
| Tele-recruitment: ~10% call-to-donation conversion | PMC 2014 India | Phone effective despite low conversion |
| Female deferral (anemia) major barrier | Marwaha 2012 | Deferral-aware scheduling critical |
| Festival drives (Oct–Dec, Holi/Dussehra) | Operational pattern | Seasonal campaign calendar |

---

## Urban vs Rural Differences

| Dimension | Urban | Rural |
|-----------|-------|-------|
| Access | Walk-in + corporate camps | Mobile camps essential |
| Communication | SMS + WhatsApp viable | SMS + voice call preferred |
| Motivation | Convenience, corporate CSR | Community identity, local leaders |
| Barriers | Time poverty | Distance, awareness gaps |

**Strategy:** Segment by `camp_ratio` and add geospatial distance when data available.

---

## Intervention Ranking Matrix

Scored 1–5 (5 = best). **Expected Retention Improvement** based on literature meta-estimates.

| Intervention | Retention Lift | Impl. Cost | Ease of Deploy | Priority Score | Evidence |
|--------------|:--------------:|:----------:|:--------------:|:--------------:|----------|
| Post-FT counseling (7-day) | ★★★★★ | Low | ★★★★★ | **1** | Bagot 2016; Mass counseling India 2014 |
| Phone call (lapsed >12 mo) | ★★★★★ | High | ★★★☆☆ | **2** | Yang 2020; Indian tele-recruitment |
| SMS altruistic reminder | ★★★★☆ | Low | ★★★★★ | **3** | Mohammed 2025; TEXT study |
| Local camp notification | ★★★★☆ | Low | ★★★★☆ | **4** | Srivastava 2025 |
| Deferral-aware scheduling (female) | ★★★★☆ | Low | ★★★★☆ | **5** | Marwaha 2012 |
| Recognition certificate (5+ donations) | ★★★☆☆ | Medium | ★★★★★ | **6** | Malhotra 2026 |
| WhatsApp group + reminders | ★★★☆☆ | Low | ★★★★★ | **7** | IAMR 2015; mixed RCT |
| Personalized blood-use message | ★★★☆☆ | Medium | ★★★☆☆ | **8** | TEXT study; Veldhuizen 2019 |
| College red ribbon club integration | ★★★☆☆ | Medium | ★★★☆☆ | **9** | Shimla study |
| Corporate CSR camp re-invite | ★★★☆☆ | Medium | ★★★★☆ | **10** | Operational best practice |

---

## Deployment Framework

```
┌─────────────────────────────────────────────────────────────┐
│                    DONOR SEGMENT                            │
├─────────────┬───────────────┬──────────────┬────────────────┤
│ First-Time  │ Active Regular│ At-Risk      │ Lapsed (>365d) │
├─────────────┼───────────────┼──────────────┼────────────────┤
│ Counseling  │ Standard SMS  │ Camp invite  │ Phone outreach │
│ Thank-you   │ cadence       │ + personalized│ + reactivation│
│ 7-day call  │ Recognition   │ SMS          │ message        │
└─────────────┴───────────────┴──────────────┴────────────────┘
```

### Festival Calendar Integration

- **Pre-position camps:** Sep (pre-Dussehra), Feb (pre-Holi), Jun (World Donor Day)
- **Reduce outreach:** Major holiday weeks (low response expected)
- **Feature:** `is_festival_season` in v2 feature store

### WhatsApp vs SMS Decision Tree

```
IF age < 35 AND smartphone opt-in → WhatsApp
ELIF long-inactive (>365d) → Phone
ELIF first-time donor → SMS (blood-used notification, TEXT study)
ELSE → SMS (lower cost, proven RCT evidence)
```

---

## Compliance (India)

- **DPDP Act 2023:** Consent for SMS/WhatsApp; hash PII in transit
- **NACO guidelines:** Dedicated counselor + recruitment officer per bank
- **Drugs & Cosmetics Act:** No monetary incentives for blood donation
