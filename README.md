# Predicting High-Cost Medicare Patients

**[→ Interactive Tableau Dashboard](https://public.tableau.com/app/profile/jialin.jiang4317/viz/MedicareAnalysis_17861454891190/MedicarePopulationOverview)** · **[→ Full Analysis Write-Up](./ANALYSIS.md)**

## Executive Summary

This project uses the CMS DE-SynPUF 2008–2010 Medicare claims dataset to investigate healthcare utilization, cost burden, and future high-cost risk among Medicare beneficiaries. The analysis spans data cleaning, patient-year feature engineering, exploratory analysis, predictive modeling, model explainability, and risk stratification — extending a capstone proposal from CPH 100: Foundations for Computational Precision Health at UC Berkeley.

Three major findings emerged:

1. **Healthcare utilization does not translate directly into cost.** Carrier and prescription drug utilization tracked payer spending fairly directly, but outpatient utilization did not — suggesting service intensity and composition matter more than encounter volume alone, particularly for outpatient care.
2. **High-cost beneficiaries represent distinct financial-burden profiles.** Populations with high prevalence of both payer spending and OOP burden consistently showed the greatest disease burden and cost across service categories, while populations with high OOP burden but lower payer spending stood out for prescription drug costs specifically — high payer spending and high patient burden appear to be overlapping but distinct risk phenotypes.
3. **Future high-cost risk is partially predictable from historical claims.** Model performance improved substantially once historical utilization/spending became available (best ROC-AUC: 0.90), but the models were better at *ranking relative risk* than *selectively identifying* all future high-cost beneficiaries — a meaningful limitation for highly targeted intervention.

Model explainability further showed that prior utilization and spending — particularly carrier claims, carrier expenditures, outpatient claims, and prescription drug expenditures — were among the strongest predictors of future high-cost status, alongside ESRD status.

Overall, claims-based analytics here supports **population-level risk stratification and early screening**, but more detailed longitudinal utilization and service-intensity features would likely be needed for high-precision intervention targeting.

## Project Background

The original proposal aimed to use the SyH-DR 2016 All-Payer Claims Dataset, but access restrictions made it impractical for a course-based project — a common bottleneck in healthcare analytics. The project was re-scoped around CMS DE-SynPUF, which preserves the structure of real Medicare claims data while being publicly accessible and reproducible.

**Dataset:** CMS DE-SynPUF (Beneficiary, Carrier, Inpatient, Outpatient, PDE) · Cohort restricted to beneficiaries with records spanning 2008–2010 · 332,694 patient-year records across 110,898 unique beneficiaries

## Key Findings

### 1. Healthcare utilization does not translate directly into cost

Across age groups, beneficiaries interacted with broadly similar types of healthcare services, yet payer costs increased substantially with age, peaking at 85–94 for inpatient and carrier costs. The utilization-cost relationship also differed by service type: carrier and prescription drug utilization tracked payer spending fairly directly, while outpatient utilization volume did not consistently correspond to cost — suggesting claim intensity and composition matter more than encounter volume, particularly for outpatient care.

> **Deep dive:** [Cost by Age](./ANALYSIS.md#cost-by-age) · [Utilization vs. Cost](./ANALYSIS.md#utilization-vs-cost)

### 2. High-cost beneficiaries exhibit distinct cost-burden profiles

High-cost status was defined independently per year using the 90th percentile of payer and OOP spending. Four cost-burden profiles emerged — high-both, high-OOP/lower-payer, high-payer/lower-OOP, and low-both. The **high-burden-both** populations consistently showed the greatest disease burden and were among the highest-cost across inpatient, outpatient, and carrier services. The **high-OOP/lower-payer** group stood out for relatively high prescription drug spending, suggesting patient financial burden can arise through different mechanisms than payer spending.

> **Deep dive:** [Cost-Burden Profiles Across Population Subgroups](./ANALYSIS.md#cost-burden-profiles-across-population-subgroups) · [Cost Composition by Cost-Burden Profile](./ANALYSIS.md#cost-composition-by-cost-burden-profile)

### 3. Disease burden is particularly associated with OOP burden

Across diabetes, cancer, CKD, COPD, ESRD, and heart failure, disease prevalence followed a consistent ordering: **high-burden-both → high-OOP/lower-payer → low-burden-both → high-payer/lower-OOP** — suggesting clinical complexity and patient financial burden are closely related, though causality can't be established from this data. ESRD was particularly notable: ~7.86% of patient-year records, but ~27% of high payer-cost and high OOP-cost records, with ~3x the average cost of non-ESRD beneficiaries.

> **Deep dive:** [Disease Burden and Cost-Burden Profiles](./ANALYSIS.md#disease-burden-and-cost-burden-profiles) · [ESRD Is Strongly Associated With High Healthcare Costs](./ANALYSIS.md#esrd-is-strongly-associated-with-high-healthcare-costs)

### 4. Future high-cost risk is partially predictable

| Model | Primary strength | Best ROC-AUC |
|---|---|---:|
| Logistic Regression | High recall / interpretable screening | 0.89 |
| Random Forest | Precision-recall middle ground | 0.90 |
| XGBoost | Discrimination + nonlinear interpretation | 0.90 |

The strongest performance occurred once historical claims became available — 2009 High OOP Burden models reached ROC-AUC 0.89–0.90, versus ~0.67–0.71 for the 2008 baseline. Operating points differed by objective: logistic regression favored recall (broad screening), random forest offered a middle ground, and XGBoost achieved strong discrimination and high precision but captured fewer actual high-cost beneficiaries at its default threshold.

> **Deep dive:** [Model Comparison](./ANALYSIS.md#model-comparison)

### 5. Historical utilization is one of the strongest predictors

SHAP analysis of XGBoost identified the strongest predictors of future high-cost status: number of carrier claims, carrier payer expenditures, number of outpatient claims, prescription drug expenditures, outpatient payer expenditures, ESRD status, and partial-year Medicare coverage. The nonlinear relationship between outpatient claim count and predicted risk reinforces the earlier finding that outpatient utilization doesn't map proportionally to cost.

> **Deep dive:** [What Drives High-Cost Predictions?](./ANALYSIS.md#what-drives-high-cost-predictions)

## Risk Stratification

Beneficiaries were ranked into five equal-sized predicted-risk quintiles rather than relying solely on binary classification. Observed event rates increased monotonically from Q1 to Q5 across models — e.g., the 2008 High OOP Burden model rose from 5.3% (Q1) to 21.4% (Q5), a 4.04x ratio. Treating the top quintile as an intervention population achieved ~25.5% precision and 51.1% recall — capturing about half of actual high-cost beneficiaries while targeting only 20% of the population. This indicates the models are more effective for relative risk ranking and low-cost screening than for highly selective intervention targeting.

> **Deep dive:** [Risk Stratification](./ANALYSIS.md#risk-stratification)

## Interactive Dashboard

**[View the Medicare Analysis Dashboard on Tableau Public](https://public.tableau.com/app/profile/jialin.jiang4317/viz/MedicareAnalysis_17861454891190/MedicarePopulationOverview)**

- **Medicare Population Overview** — demographics, spending patterns, service utilization, geographic variation
- **Beneficiary Cost-Burden Segmentation** — high payer spending, high OOP burden, cost composition, utilization, demographic differences

## Repository Structure

| Notebook | Focus |
|---|---|
| `0_data_loading.ipynb` | Source data loading |
| `1_data_cleaning.ipynb` | Data auditing, cleaning, validation, feature engineering |
| `2_eda_and_feature_engineering.ipynb` | Population, utilization, cost, disease burden |
| `3_modeling.ipynb` | Logistic regression, random forest, XGBoost |
| `4_analyzing.ipynb` | Feature importance, SHAP, risk stratification |

## Dataset & Limitations

- The dataset is **synthetic** (CMS DE-SynPUF) and does not represent actual individual beneficiaries
- Covers **2008–2010** — spending patterns and the healthcare environment are historical
- Cohort is restricted to beneficiaries with records spanning all three years
- High-cost thresholds are defined relative to the annual sample, not universal clinical/financial thresholds
- Demographic representation is imbalanced, particularly across racial and ethnic groups (see [Population Demographics](./ANALYSIS.md#population-demographics))
- Predictive associations should not be interpreted as causal
- The observed 2010 spending decline requires further investigation and shouldn't be attributed to a specific policy based on this analysis alone

## Software Environment

Python 3.13.5 · Scikit-learn 1.6.1 · XGBoost 3.1.2 · Tableau


