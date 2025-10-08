# SDOH-Driven Readmission Risk Prediction
### Who Needs a Patient Navigator Most? A Machine Learning Approach

**Author:** Sai Manasa Adduru, MPH (Epidemiology), PharmD
**Methods:** Gradient Boosting · Logistic Regression · ROC/AUC · Risk Stratification · SDOH Feature Engineering

---

## Overview

This project builds a **30-day readmission risk prediction model** that combines clinical features with **Social Determinants of Health (SDOH)** to identify which patients should be prioritized for patient navigation services.

The core research question: *Can adding SDOH data (housing instability, food insecurity, transportation barriers, social isolation) improve our ability to predict who will be readmitted — and who needs an advocate most?*

---

## Key Findings

| Metric | Value |
|--------|-------|
| Clinical-only model AUC | 0.610 |
| Clinical + SDOH model AUC | **0.642** |
| GBM model (5-fold CV AUC) | 0.605 ± 0.013 |
| Critical-tier patients (top 5%) | **88.7% actual readmission rate** |
| SDOH burden 0 vs 5+ readmission | 13.8% → 31.1% |

---

## Navigation Priority Tiers

| Tier | N | Actual Readmit Rate |
|------|---|---------------------|
| **Critical** | 231 (5%) | **88.7%** |
| High | 884 (18%) | 47.2% |
| Moderate | 2,440 (49%) | 17.9% |
| Low | 1,445 (29%) | 6.3% |

The model correctly identifies the highest-risk 5% of patients with 88.7% readmission rate — enabling targeted navigation deployment.

---

## Why SDOH Matters

Top features by model importance:
1. **Area Deprivation Index** [SDOH] — 23.4%
2. **Age** [Clinical] — 22.1%
3. **Index LOS** [Clinical] — 10.1%
4. **SDOH Burden Score** [SDOH] — 8.3%
5. **Prior Admissions** [Clinical] — 7.7%

SDOH features account for **~40% of total model importance** — demonstrating they are essential, not supplementary, for accurate risk prediction.

---

## Repository Structure

```
├── data/
│   ├── simulate_patients.py       # 5,000-patient synthetic EHR + SDOH dataset
│   └── sdoh_patient_cohort.csv    # Generated dataset
├── analysis/
│   └── risk_model.py              # GBM model, ROC curves, risk stratification
└── output/figures/                # 4 publication-quality figures
```

---

## Reproduce

```bash
git clone https://github.com/saimadduru/sdoh-readmission-risk
cd sdoh-readmission-risk
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python3 data/simulate_patients.py
python3 analysis/risk_model.py
```

---

`Python 3.9` · `pandas` · `scikit-learn` · `matplotlib` · `seaborn`
