"""
Synthetic patient dataset with Social Determinants of Health (SDOH) features.
Simulates an EHR + claims + SDOH-linked dataset for readmission risk modeling.
N=5,000 patients discharged from acute care.
"""

import numpy as np
import pandas as pd

np.random.seed(7)
N = 5000

# Clinical features
age             = np.random.normal(63, 14, N).clip(18, 92).astype(int)
female          = np.random.binomial(1, 0.51, N)
cci             = np.random.poisson(2.2, N).clip(0, 9)
index_los       = np.random.poisson(4.1, N).clip(1, 21)
prior_admits_12m= np.random.poisson(1.3, N).clip(0, 7)
chf             = np.random.binomial(1, 0.29, N)
copd            = np.random.binomial(1, 0.25, N)
diabetes        = np.random.binomial(1, 0.37, N)
ckd             = np.random.binomial(1, 0.22, N)
discharge_to_snf= np.random.binomial(1, 0.18, N)  # Skilled nursing facility

# SDOH features — the key differentiator of this project
housing_instability  = np.random.binomial(1, 0.19, N)   # eviction risk / unstable housing
food_insecurity      = np.random.binomial(1, 0.24, N)
transportation_barrier= np.random.binomial(1, 0.21, N)
social_isolation     = np.random.binomial(1, 0.28, N)   # lives alone, no caregiver
low_health_literacy  = np.random.binomial(1, 0.32, N)
uninsured_gap        = np.random.binomial(1, 0.13, N)   # coverage gap
area_deprivation     = np.clip(np.random.normal(55, 20, N), 1, 99).astype(int)  # ADI score
dual_eligible        = np.random.binomial(1, 0.27, N)

# SDOH composite score (0–8)
sdoh_burden = (housing_instability + food_insecurity + transportation_barrier +
               social_isolation + low_health_literacy + uninsured_gap +
               (area_deprivation > 70).astype(int) + dual_eligible)

# Navigation assignment — higher SDOH burden = more likely to be navigated
nav_logit = (
    -1.5
    + 0.4  * (sdoh_burden >= 3).astype(int)
    + 0.3  * dual_eligible
    + 0.25 * (prior_admits_12m >= 2).astype(int)
    + 0.2  * chf
    + np.random.normal(0, 0.5, N)
)
navigated = np.random.binomial(1, 1/(1+np.exp(-nav_logit)), N)

# 30-day readmission — SDOH features AND navigation both matter
readmit_logit = (
    -2.1
    + 0.022 * (age - 63)
    + 0.5  * chf
    + 0.4  * copd
    + 0.3  * ckd
    + 0.35 * (prior_admits_12m >= 2).astype(int)
    + 0.4  * housing_instability
    + 0.35 * food_insecurity
    + 0.3  * transportation_barrier
    + 0.4  * social_isolation
    + 0.25 * low_health_literacy
    + 0.3  * (area_deprivation > 70).astype(int)
    - 0.5  * navigated
    + np.random.normal(0, 0.4, N)
)
readmit_30d = np.random.binomial(1, 1/(1+np.exp(-readmit_logit)), N)

# Navigation need score (ground truth for supervised model target)
nav_need = sdoh_burden + cci + prior_admits_12m + chf + copd

df = pd.DataFrame({
    "patient_id":              [f"P{str(i).zfill(5)}" for i in range(N)],
    "age":                     age,
    "female":                  female,
    "cci":                     cci,
    "index_los":               index_los,
    "prior_admits_12m":        prior_admits_12m,
    "chf":                     chf,
    "copd":                    copd,
    "diabetes":                diabetes,
    "ckd":                     ckd,
    "discharge_to_snf":        discharge_to_snf,
    "housing_instability":     housing_instability,
    "food_insecurity":         food_insecurity,
    "transportation_barrier":  transportation_barrier,
    "social_isolation":        social_isolation,
    "low_health_literacy":     low_health_literacy,
    "uninsured_gap":           uninsured_gap,
    "area_deprivation_index":  area_deprivation,
    "dual_eligible":           dual_eligible,
    "sdoh_burden_score":       sdoh_burden,
    "navigated":               navigated,
    "readmit_30d":             readmit_30d,
})

df.to_csv("data/sdoh_patient_cohort.csv", index=False)
print(f"Generated {N:,} patients")
print(f"30-day readmission rate: {readmit_30d.mean():.1%}")
print(f"Mean SDOH burden score:  {sdoh_burden.mean():.2f}/8")
print(f"Patients with SDOH >= 3: {(sdoh_burden>=3).sum():,} ({(sdoh_burden>=3).mean():.0%})")
