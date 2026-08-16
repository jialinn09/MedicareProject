# Predicting High-Cost Medicare Patients

# Project Background

This project builds upon my capstone proposal for **CPH 100: Foundations for Computational Precision Health** at the University of California, Berkeley. I designed an end-to-end healthcare analytics pipeline using administrative claims data to find patients with high future healthcare costs. The project covers the full workflow, from data cleaning and feature engineering to predictive modeling, explainability analysis, and risk stratification.

## Dataset Selection & Motivation

The original project proposal aimed to use the **SyH-DR 2016 All-Payer Claims Dataset** to study spending patterns across Medicare, Medicaid, and commercial insurance populations. However, access restrictions and data governance requirements made the dataset impractical for a course-based project.

This challenge highlighted a common bottleneck in healthcare analytics: obtaining access to real-world patient and payer data.

To address this limitation, the project was re-scoped using the **Centers for Medicare & Medicaid Services (CMS) DE-SynPUF 2008–2010 dataset**. Although synthetic, DE-SynPUF preserves the structure and relationships found in real Medicare claims data, making it well suited for developing reproducible healthcare analytics workflows.

While the dataset is dated, the analytical framework remains highly relevant for modern healthcare cost prediction, risk stratification, and population health management tasks.

### Software Environment

* Python 3.13.5
* Scikit-learn 1.6.1
* XGBoost 3.1.2

## Interactive Dashboard

An accompanying Tableau dashboard is available here:

https://public.tableau.com/app/profile/jialin.jiang4317/viz/MedicareAnalysis_17861454891190/MedicarePopulationOverview

The dashboard is organized into two modules:

### 1. Medicare Population Overview

Provides population-level insights into:

* Demographic characteristics
* Healthcare spending patterns
* Service utilization trends
* Geographic variation

### 2. Beneficiary Cost Burden Segmentation

Beneficiaries are segmented according to the prevalence of:

* High out-of-pocket (OOP) burden
* High payer spending

For each calendar year, beneficiaries exceeding the **90th percentile** of spending are classified as high-cost.

Population subgroups are categorized into four cost-burden profiles:

1. High OOP burden prevalence, lower payer spending
2. High payer spending prevalence, lower OOP burden
3. High prevalence of both payer spending and OOP burden
4. Low prevalence of both

A subgroup is considered to have high prevalence when more than 10% of beneficiaries fall within the corresponding high-cost category. Each cost-burden profile includes:

* Cost composition analyses
* Utilization summaries
* Demographic breakdowns

to help identify structural differences across beneficiary populations.

## Repository Structure

### 0_data_loading.ipynb

Loads the five CMS source datasets:

* Beneficiary
* Carrier
* Inpatient
* Outpatient
* PDE (Prescription Drug Event)

### 1_data_cleaning.ipynb

Performs data auditing, cleaning, validation, and feature engineering.

The cohort is restricted to beneficiaries with records spanning all three years (2008–2010). All datasets are transformed into patient-year level records to support longitudinal spending analyses.

Key cleaning activities include:

#### Beneficiary

* Missing value and duplicate assessment
* Mapping coded demographic variables (race, state, etc.) to readable values
* Standardizing binary encodings
* Consistency checks across dates and demographic fields
* Age-band generation for downstream analysis
* Column pruning and renaming

#### Carrier

* Claim-level validation checks
* Payment logic verification
* Disease category and chronic condition mapping using ICD codes
* Patient-year aggregation
* Column pruning and renaming

#### PDE

* Patient-year aggregation of prescription utilization and spending metrics

#### Inpatient & Outpatient

* Claim-level validation
* Spending consistency checks
* Patient-year feature generation

### 2_eda_and_feature_engineering.ipynb

Performs exploratory analysis and develops predictive features.

Major tasks include:

* Validation that claim-level payments reconcile with beneficiary-level totals
* Construction of spending measures:

  * Payer cost
  * Out-of-pocket cost
  * Claim processing burden
* Definition of high-cost beneficiaries using the 90th percentile threshold
* Investigation of ESRD populations and incorporation of ESRD status as a predictive feature

### 3_modeling.ipynb

Develops and evaluates:

* Logistic Regression
* Random Forest
* XGBoost

Separate models are trained for:

* High Out-of-Pocket Burden
* High Payer Cost

The feature space expands over time:

* 2008 models rely primarily on demographics and ESRD status
* 2009 models incorporate 2008 historical utilization and spending information
* 2010 models incorporate cumulative historical records from prior years

### 4_analyzing.ipynb

Performs model interpretation and comparative analysis.

Major analyses include:

* Feature importance extraction
* Feature family aggregation
* SHAP explainability analysis
* Risk stratification analysis using predicted-risk quintiles
* Comparison of predictive drivers across years and model types

The goal is to understand not only which beneficiaries are likely to become high-cost, but also the underlying factors driving those predictions.

# Insights

The analysis is organized around two stakeholder perspectives: **Health Analytics** and **Payer Strategy**.

For the **Health Analytics team**, several questions guide the analysis:

1. What does the Medicare population look like from 2008 to 2010?
2. What factors are associated with healthcare utilization and cost?
3. Who are the highest-cost beneficiaries?

For the **Payer Strategy team**, the analysis focuses on:

1. Can we identify beneficiaries who are likely to become high-cost?
2. What factors are associated with high healthcare costs?
3. Where might interventions be targeted, and why?

The following sections examine these questions through population-level EDA, utilization patterns, cost drivers, and high-cost beneficiary characteristics.

## #1: What Does the Medicare Population Look Like from 2008 to 2010?

### Population Demographics

After integrating the cleaned administrative claims datasets, the final dataset contains **332,694 patient-year records** from 2008–2010. Because the unit of analysis is a patient-year, a beneficiary can contribute multiple records across different years.

The demographic analysis examines each patient-year record by **sex, age band, and race**. The heatmap uses the average number of distinct service types accessed by each population as its color gradient, where darker cells represent lower average service-type utilization.

### Key Insight 1: The dataset is disproportionately represented by White beneficiaries and women
<img width="786" height="330" alt="Screenshot 2026-08-15 at 1 55 11 PM" src="https://github.com/user-attachments/assets/c4a9fa06-3c88-4df3-a182-f2670ecfaf83" />
Women account for **55.57%** of patient-year records, compared with **44.43%** for men. The dataset is also predominantly White:

* White women: 58,628 records (**82.76% of female records**)
* White men: 47,079 records (**83.10% of male records**)
* Black women: 7,682 records (**10.80% of female records**)
* Black men: 5,695 records (**10.10% of male records**)
* Hispanic women: 1,578 records (**2.23% of female records**)
* Hispanic men: 1,240 records (**2.20% of male records**)
* Other-race women: 2,950 records (**4.16% of female records**)
* Other-race men: 2,624 records (**4.60% of male records**)

This demographic imbalance is an important limitation when interpreting downstream findings. Patterns identified in this dataset may be more representative of the White Medicare population than of racial and ethnic groups that are less represented in the sample.

### Key Insight 2: Most beneficiaries interact with multiple service types
Across demographic groups, the average number of distinct service types accessed ranges from approximately **1.6 to 2.6**, suggesting that most patient-year records involve interaction with roughly two to three types of healthcare services.

Across the full population (not shown on the visualization but available at 1_data_cleaning.ipynb):

* **41.30%** accessed 3 distinct service types
* **31.04%** accessed 2 distinct service types
* **17.26%** accessed 1 distinct service type
* **10.40%** accessed all 4 service types

The populations with the lowest average service-type utilization include several of the oldest age groups and selected non-White populations. For example, Black men aged 95+ contributed only 101 patient-year records, while other-race men aged 95+ contributed 45 records.

These smaller population cells should be interpreted cautiously. Lower observed service-type utilization could reflect differences in disease burden or healthcare access, but it may also be influenced by **small sample sizes and the structure of the underlying claims data**. 

### Service-Type Utilization
<img width="607" height="235" alt="Screenshot 2026-08-15 at 1 53 35 PM" src="https://github.com/user-attachments/assets/b8fac1e8-b179-474c-94b4-27474a5ed96e" />

The distribution of service combinations provides additional context for how beneficiaries interact with the healthcare system.

The largest utilization pattern consists of **carrier, outpatient, and prescription drug services**, representing approximately **37% of patient-year records**. The second-largest pattern consists of **carrier and prescription drug services** at approximately **18.8%**, followed by patient-year records with **prescription drug utilization only** at approximately **11.3%**.

Overall, this suggests that prescription drug and physician/carrier services are central components of healthcare utilization within the population, while inpatient services are used by a smaller subset of beneficiaries.

### Utilization by Age
<img width="854" height="540" alt="Screenshot 2026-08-15 at 9 27 51 PM" src="https://github.com/user-attachments/assets/b3a3dd6a-2c92-462f-9135-d1b1414ebf50" />

Service utilization patterns remain broadly consistent across age bands, with **carrier and prescription drug services accounting for the largest shares of utilization**.

One notable exception is the **under-65 Medicare population**, which has relatively high prescription drug utilization. This population likely differs clinically from the traditional Medicare population because beneficiaries under 65 generally qualify for Medicare through disability or specific qualifying conditions. However, the dataset alone cannot establish that disability-related disease complexity is the cause of the higher prescription drug utilization.

The higher prescription drug utilization is accompanied by higher drug-related payer costs in the <65 population, suggesting that this subgroup warrants additional investigation when examining cost drivers.

### Cost by Age
<img width="869" height="522" alt="Screenshot 2026-08-15 at 9 29 21 PM" src="https://github.com/user-attachments/assets/0a5c1dd7-42f3-4cca-865d-57d1e0c0c6e9" />

Although the overall distribution of service types is relatively stable across age groups, **payer costs increase with age for several service categories**.

In particular:

* Inpatient payer costs increase with age and peak among beneficiaries aged 85–94.
* Carrier payer costs also increase with age and peak among beneficiaries aged 85–94.
* Prescription drug costs generally increase with age.
* Other categories remain comparatively stable across age bands.

This divergence between **relatively stable utilization patterns and increasing costs** suggests that utilization volume alone does not fully explain healthcare spending. Among older beneficiaries, increasing clinical complexity and intensity of care may contribute to higher costs even when the number of distinct service types remains similar.

This finding motivates the later analysis of chronic conditions and high-cost beneficiaries.

## #2: Who Are the High-Cost Beneficiaries?

To identify high-cost beneficiaries, payer and patient out-of-pocket (OOP) costs were examined separately. Because healthcare spending varies substantially by year, the high-cost threshold was defined independently for each year using the **top 10% of annual spending**.

| Year | Top 10% Payer Cost Cutoff | Top 10% OOP Cost Cutoff |
| ---- | ------------------------: | ----------------------: |
| 2008 |                   $12,520 |                  $2,544 |
| 2009 |                   $13,530 |                  $2,578 |
| 2010 |                    $7,340 |                  $1,590 |

This approach avoids applying a single dollar threshold across years with different spending distributions and allows high-cost beneficiaries to be identified relative to their contemporaneous population.

### ESRD Is Strongly Associated With High Healthcare Costs

ESRD represents approximately **7.86% of patient-year records**, yet it is disproportionately represented among high-cost beneficiaries.

<img width="645" height="182" alt="Screenshot 2026-08-15 at 1 58 39 PM" src="https://github.com/user-attachments/assets/aabf314d-4cd3-4403-9f69-f2739e085084" />

Among patient-year records classified as high payer-cost (not shown on the visualization but available at 2_eda_and_feature_engineering.ipynb):

* **72.7%** are non-ESRD
* **27.3%** are ESRD

Among high OOP-cost patient-year records:

* **71.37%** are non-ESRD
* **28.63%** are ESRD

The contrast becomes even more apparent when examining average costs and utilization: **ESRD beneficiaries have approximately three times the average payer cost and OOP cost of non-ESRD beneficiaries**, alongside higher healthcare utilization.

From a payer strategy perspective, this makes ESRD an important clinical indicator for identifying populations with disproportionately high resource utilization and financial burden.

### The 2010 Cost Decline Requires Further Investigation

One of the most notable patterns is a sharp decline in observed costs in 2010 for the same cohort of approximately **110,898 beneficiaries**.

This finding should be treated as an **observation rather than a causal conclusion**. Several explanations are possible, including changes in utilization, claims patterns, reimbursement, coding, or the composition of services captured in the dataset.

External Medicare research provides some useful context. The Commonwealth Fund found that Medicare spending growth slowed during this broader period and attributed reductions to a combination of changes in chronic-condition spending, service utilization, and evolving approaches to care delivery and payment. Cardiovascular disease spending, in particular, declined substantially during the 2007–2010 to 2011–2014 comparison period.

However, because this project uses a specific **DE-SynPUF sample and only covers 2008–2010**, the observed 2010 decline cannot be attributed directly to the Affordable Care Act or to changes in heart attack and stroke prevalence without additional analysis. The decline should therefore be considered a finding that warrants further investigation rather than evidence of a specific policy effect.

### Cost-Burden Profiles Across Population Subgroups

To further characterize high-cost populations, patient-year records were grouped into four cost-burden profiles based on the prevalence of high payer spending and high out-of-pocket (OOP) spending within each demographic subgroup. A subgroup was classified as having **high prevalence** when more than 10% of its patient-year records fell within the corresponding high-cost category.

This creates four cost-burden profiles:

| Cost-burden profile                      | Interpretation                                                            | Population subgroups                                                 |
| ---------------------------------------- | ------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| 🟧 High OOP burden, lower payer spending | High prevalence of high OOP cost, but lower prevalence of high payer cost | Hispanic females 75–84; Hispanic males 85–94                         |
| 🟨 High payer spending, lower OOP burden | High prevalence of high payer cost, but lower prevalence of high OOP cost | Black males 95+                                                      |
| 🟥 High prevalence of both               | High prevalence of both high payer and high OOP costs                     | Females 85–94 and <65; non-Hispanic males 85–94; non-Black males 95+ |
| 🟩 Low prevalence of both                | Lower prevalence of both high payer and high OOP costs                    | Ages 65–74 across demographic groups                                 |

<img width="764" height="540" alt="Screenshot 2026-08-15 at 9 37 55 PM" src="https://github.com/user-attachments/assets/28fb81a0-9d73-4067-8596-61a53dc86d9e" />

The concentration of the **high-burden-both** profile among females aged 85–94 and <65, as well as males aged 85–94 and 95+, suggests that certain age groups experience substantial financial burden from both the payer and patient perspectives.

The high-burden profiles also show some consistency with utilization patterns observed earlier. For example, the Black male 95+ subgroup in the high-payer/low-OOP profile and the non-Black male 95+ subgroup in the high-both profile both exhibit relatively low average distinct service-type utilization. This suggests that **the number of distinct service types alone may not adequately capture cost intensity**. A beneficiary may interact with fewer types of services while still generating substantially higher costs.

Similarly, the low-burden-both profile is concentrated among beneficiaries aged 65–74, which is consistent with the relatively lower service-type utilization observed in some of these populations earlier in the analysis.

### Cost Composition by Cost-Burden Profile

Breaking total costs down by service type reveals that different high-cost populations may experience different mechanisms of financial burden.

**Inpatient costs** are highest among populations with a high prevalence of both payer and OOP spending, followed by populations with high payer spending but lower OOP burden. The same pattern is observed for **outpatient payer costs**.

<img width="764" height="546" alt="ip insurance" src="https://github.com/user-attachments/assets/54139a1e-76b4-4882-852b-93662920bc01" />

This suggests that inpatient and outpatient services are major contributors to the high payer spending observed among the highest-burden populations.

**Carrier payer costs** show a similar pattern, with the high-burden-both group having the highest average cost. However, the high-OOP/lower-payer group and low-burden-both group are nearly tied for the second-highest carrier costs.

<img width="764" height="546" alt="carrier insurance" src="https://github.com/user-attachments/assets/df8f3ae0-94af-4c31-ab95-c6b21421ad8a" />

The pattern differs for **prescription drug (PDE) payer costs**. The high-OOP/lower-payer group has the highest average prescription drug payer cost, exceeding the high-burden-both group by approximately $500.

<img width="764" height="546" alt="pde insurance" src="https://github.com/user-attachments/assets/0ff15251-791f-4fc9-ac55-da2ba70a2f57" />

This divergence is particularly notable because the high-OOP/lower-payer group is defined by substantial patient financial burden despite relatively lower prevalence of high payer spending. One possible explanation is **higher prescription drug utilization or greater patient cost-sharing**, where prescription-related expenses contribute disproportionately to patient OOP burden relative to other service categories. Additional analysis of prescription volume, drug spending, and patient cost-sharing would be needed to distinguish between these mechanisms.

### Key Takeaway

High-cost Medicare beneficiaries are not a homogeneous population. Different demographic groups can arrive at high overall healthcare costs through **different combinations of payer spending and patient financial burden**.

The high-burden-both populations represent the most consistently expensive profile across inpatient, outpatient, and carrier services, while the high-OOP/lower-payer profile stands out for its relatively high prescription drug costs.

For payer strategy, this distinction matters because **intervention strategies may need to differ depending on whether the primary burden comes from high medical utilization, prescription drug spending, patient cost-sharing, or a combination of these factors.**

### Utilization vs. Cost

Examining utilization alongside payer costs reveals different relationships across service types.

For **inpatient services**, utilization follows the expected pattern: the high-prevalence-of-both group has the highest average utilization, followed by the high-payer-spending/lower-OOP group. Overall, higher inpatient utilization is associated with higher payer costs. However, the difference in average utilization between the two highest-utilization groups is relatively small, while their average payer cost burden differs by approximately **$300**. This suggests that utilization volume alone may not fully explain inpatient cost differences and warrants further investigation into differences in service intensity and claim composition.

<img width="590" height="546" alt="IP util" src="https://github.com/user-attachments/assets/21298c89-5f48-4e7c-8f25-475e93b66d2e" />

The relationship is less consistent for **outpatient services**. Average outpatient payer costs follow the expected descending order:

**High prevalence of both → High payer spending/lower OOP → High OOP/lower payer → Low prevalence of both.**

However, the utilization ranking does not follow the same pattern. The high-OOP/lower-payer and low-burden-both groups exhibit higher average outpatient utilization than the high-payer/lower-OOP group despite having lower average payer costs.

<img width="590" height="546" alt="OP util" src="https://github.com/user-attachments/assets/4fce10ba-9a30-4e4e-a4cd-2ad1843289d2" />

This divergence suggests that **outpatient utilization volume is not a sufficient proxy for outpatient payer burden**. Some groups may have more outpatient encounters but lower-cost encounters, while others may have fewer encounters with substantially higher costs per encounter.

This makes outpatient claims particularly interesting for further analysis. A useful next step would be to examine **payer cost per utilization event**, as well as the underlying procedure, provider, or service mix, to determine whether differences in cost intensity explain the observed divergence.

In contrast, **carrier utilization and payer cost show a much more direct relationship**: groups with higher average carrier utilization also exhibit higher carrier payer costs. The same pattern is observed for **prescription drug (PDE) utilization**, where higher prescription utilization corresponds with higher payer spending.

<img width="590" height="546" alt="carrier util" src="https://github.com/user-attachments/assets/ee413c73-651b-4b66-991a-1b471eabaf3a" />
<img width="590" height="546" alt="pde util" src="https://github.com/user-attachments/assets/d6017098-b70b-4a6a-876a-6ac1b2f9e69f" />

Overall, these results suggest that the relationship between utilization and cost is **service-specific**. Carrier and prescription drug costs appear more closely tied to utilization volume, whereas outpatient costs may depend more heavily on the intensity and composition of individual encounters.

### Disease Burden and Cost-Burden Profiles

The relationship between clinical disease burden and cost-burden profiles was examined across six major disease categories: **diabetes, cancer, chronic kidney disease (CKD), chronic obstructive pulmonary disease (COPD), end-stage renal disease (ESRD), and heart failure**.

Although the exact prevalence varies by disease, a remarkably consistent pattern emerges across all six conditions:

**High prevalence of both payer and OOP burden → High OOP burden/lower payer burden → Low prevalence of both → High payer burden/lower OOP burden.**

In other words, the populations experiencing high patient out-of-pocket burden consistently show higher prevalence of chronic and serious diseases than populations characterized primarily by high payer spending.

### COPD as an Illustrative Example

<img width="478" height="546" alt="Screenshot 2026-08-16 at 11 39 10 AM" src="https://github.com/user-attachments/assets/d38e070e-6733-4811-89c8-8764bc61b987" />

COPD demonstrates this pattern clearly. The high-burden-both population has the highest disease prevalence, followed by the high-OOP/lower-payer population. The low-burden-both population has lower prevalence, while the high-payer/lower-OOP population consistently exhibits the lowest disease prevalence.

The same ordering is observed across diabetes, cancer, CKD, ESRD, and heart failure, suggesting that this is not isolated to a single disease category.

### Key Insight

The consistency of this pattern suggests that **clinical disease burden may be more closely associated with patient OOP burden than with payer-only cost burden** within this dataset.

One potential explanation is that patients with chronic or complex conditions may require sustained treatment across multiple services, resulting in recurring patient cost-sharing even when individual claims are not necessarily among the highest payer-cost claims. However, the current analysis cannot determine whether disease burden itself drives OOP spending or whether the relationship is mediated by utilization, treatment intensity, service mix, or cost-sharing structure.

This distinction is particularly important because it suggests that **high payer spending and high patient financial burden may represent different risk phenotypes**. A population with high payer spending but relatively low disease prevalence may be driven by a smaller number of high-intensity services, whereas populations with substantial chronic disease burden may experience more persistent healthcare needs and associated OOP expenses.

### Future Analysis

A natural next step is to connect disease prevalence with **claim utilization and claim intensity**. For example, future analysis could examine:

* Number of inpatient, outpatient, carrier, and prescription drug claims by disease
* Average and total utilization among beneficiaries with each condition
* Payer cost and OOP cost per utilization event
* Number of chronic conditions per patient-year
* Service-type combinations associated with specific diseases
* Whether disease-specific utilization predicts high OOP burden after accounting for age and other demographic characteristics

This would help distinguish between two potential mechanisms: **high OOP burden resulting from frequent, sustained healthcare utilization** versus **high OOP burden resulting from a smaller number of high-intensity services**.

More broadly, this analysis reinforces the finding that **high healthcare spending is not a single phenotype**. Clinical complexity, utilization intensity, payer spending, and patient financial burden may interact in different ways across beneficiary populations.

# #3: Can We Identify Future High-Cost Beneficiaries?

The final modeling question asks whether beneficiary characteristics, historical utilization, disease burden, and prior spending can be used to identify beneficiaries at elevated risk of becoming high-cost.

To preserve the temporal structure of the analysis:

* **2008:** Predictions use baseline beneficiary demographics and ESRD information.
* **2009:** Predictions incorporate 2008 historical information alongside beneficiary characteristics.
* **2010:** Predictions incorporate cumulative information from **2008 and 2009**, allowing the models to leverage a longer history of beneficiary utilization, disease burden, and spending.

High-cost outcomes were defined independently within each year using the top 10% of annual payer spending and OOP burden:

| Year | High Payer Cost Prevalence | High OOP Burden Prevalence |
| ---- | -------------------------: | -------------------------: |
| 2008 |                      9.99% |                      9.98% |
| 2009 |                      9.99% |                      9.97% |
| 2010 |                      9.99% |                     10.00% |

The near-10% prevalence across all outcomes is expected because high-cost beneficiaries were defined using annual top-decile thresholds.

## Model Comparison

Three classification approaches were evaluated: **logistic regression, random forest, and XGBoost**.

### Logistic Regression: High-Recall Screening

Logistic regression was configured with balanced class weighting to account for the relatively rare high-cost outcomes. This produced a model that prioritizes sensitivity to the positive class, accepting more false positives in exchange for capturing more actual high-cost beneficiaries.

| Target               | ROC-AUC | Precision | Recall |
| -------------------- | ------: | --------: | -----: |
| High Payer Cost 2008 |    0.66 |      0.25 |   0.37 |
| High OOP Burden 2008 |    0.67 |      0.27 |   0.38 |
| High Payer Cost 2009 |    0.83 |      0.24 |   0.75 |
| High OOP Burden 2009 |    0.89 |      0.32 |   0.83 |
| High Payer Cost 2010 |    0.77 |      0.19 |   0.72 |
| High OOP Burden 2010 |    0.81 |      0.22 |   0.77 |

Performance improved substantially when historical information became available. In particular, the 2009 and 2010 models achieved substantially higher recall than the 2008 baseline models, suggesting that prior beneficiary information provides useful predictive signal for future high-cost outcomes.

The tradeoff is relatively low precision, meaning that many beneficiaries flagged by the model would not ultimately fall within the top-cost group. This makes logistic regression more appropriate for **broad, low-cost screening or risk review** than for highly selective interventions.

### Random Forest: Intermediate Precision-Recall Tradeoff

Random forest was evaluated as a nonlinear benchmark and generally produced an intermediate tradeoff between precision and recall.

| Target               | ROC-AUC | Precision | Recall |
| -------------------- | ------: | --------: | -----: |
| High Payer Cost 2008 |    0.69 |      0.22 |   0.43 |
| High OOP Burden 2008 |    0.70 |      0.24 |   0.46 |
| High Payer Cost 2009 |    0.83 |      0.37 |   0.49 |
| High OOP Burden 2009 |    0.90 |      0.43 |   0.68 |
| High Payer Cost 2010 |    0.79 |      0.32 |   0.38 |
| High OOP Burden 2010 |    0.82 |      0.31 |   0.48 |

The random forest models generally improved precision relative to logistic regression while sacrificing some recall. This provides a useful middle ground when the cost of false positives becomes more important.

### XGBoost: Strongest Discriminative Performance

XGBoost achieved the strongest or near-strongest ROC-AUC across the majority of targets and was therefore selected for detailed SHAP-based interpretation.

| Target               | ROC-AUC | Precision | Recall |
| -------------------- | ------: | --------: | -----: |
| High Payer Cost 2008 |    0.69 |      0.48 |   0.03 |
| High OOP Burden 2008 |    0.71 |      0.61 |   0.03 |
| High Payer Cost 2009 |    0.84 |      0.75 |   0.15 |
| High OOP Burden 2009 |    0.90 |      0.67 |   0.33 |
| High Payer Cost 2010 |    0.79 |      0.75 |   0.07 |
| High OOP Burden 2010 |    0.83 |      0.68 |   0.08 |

XGBoost demonstrates a markedly different operating point from logistic regression. Its high precision indicates that predictions classified as positive tend to be relatively high-confidence, but its low recall means that the model identifies only a small fraction of the eventual high-cost population at the selected classification threshold.

This illustrates an important distinction: **strong discriminative performance does not necessarily translate into high-recall targeting at a fixed classification threshold.**

## What Drives High-Cost Predictions?

<img width="1055" height="592" alt="Screenshot 2026-08-16 at 12 47 45 PM" src="https://github.com/user-attachments/assets/448bc00d-749c-4175-8075-d9fcc1cef112" />

Feature-family analysis across models consistently identified **historical cost, utilization, disease burden, chronic condition burden, and demographic characteristics** as important sources of predictive information.

XGBoost SHAP analysis provided a more granular view of the individual features driving predictions. The strongest contributors included:

* Number of carrier (Part B) claims
* Total carrier payer expenditures
* Number of outpatient claims
* Prescription drug expenditures
* Outpatient payer expenditures

An example result from high OOP burden for 2009 XGB model SHAP is included below:

<img width="674" height="701" alt="Screenshot 2026-08-16 at 12 49 01 PM" src="https://github.com/user-attachments/assets/a7694d43-376c-4612-b87e-32fe8f29eb63" />

These findings reinforce several patterns identified during EDA.

First, the strong contribution of carrier utilization and prescription drug expenditures is consistent with the earlier finding that **carrier and prescription drug utilization have relatively direct relationships with payer costs**.

Second, outpatient claim count shows nonlinear effects in the XGBoost model. This is consistent with the earlier EDA finding that **outpatient utilization does not map proportionally to payer cost**. The same number of outpatient claims may therefore correspond to substantially different levels of future cost depending on the beneficiary's broader clinical and utilization profile.

### Clinical and Coverage Indicators

**ESRD** consistently emerged as an important clinical indicator of high-cost risk, reinforcing the EDA finding that ESRD beneficiaries experience substantially higher payer and OOP costs.

Partial-year Medicare coverage was also associated with higher predicted future cost. Possible explanations include new enrollment, coverage transitions, or other differences in beneficiary circumstances. Because the current analysis cannot distinguish among these mechanisms, this finding should be treated as a hypothesis for further investigation rather than a causal relationship.

## Risk Stratification

Beyond binary classification, predicted probabilities were used to rank beneficiaries into five equally sized risk groups:

* **Q1:** Lowest 20% predicted risk
* **Q2:** 20–40%
* **Q3:** 40–60%
* **Q4:** 60–80%
* **Q5:** Highest 20%

Risk stratification demonstrated **meaningful separation** across the models: quintile sizes were balanced, and observed event rates increased monotonically from Q1 to Q5.

For example, in the 2008 High OOP Burden model, the observed event rate increased from **5.3% in Q1 to 21.4% in Q5**, representing a **4.04-fold difference**. Mean predicted probability increased from 30.4% to 64.0% across the same groups.

<img width="860" height="153" alt="Screenshot 2026-08-16 at 12 46 18 PM" src="https://github.com/user-attachments/assets/7a82a8d8-1dca-453d-a223-25fdf604b383" />

Similar monotonic patterns were observed across the year-outcome models, indicating that the models can meaningfully **rank beneficiaries according to relative risk**, even when binary classification performance is more limited.

When the highest-risk quintile was treated as the intervention population, it achieved **25.5% precision and 51.1% recall**. In other words, targeting the top 20% of predicted-risk beneficiaries captured approximately half of the actual high-cost population, while approximately one in four targeted beneficiaries ultimately belonged to the high-cost group.

This represents meaningful risk concentration, but also demonstrates the limitations of highly selective targeting: **a substantial proportion of high-cost beneficiaries remain outside the highest-risk quintile**.

For a payer with relatively low-cost screening or review resources, this risk stratification approach could still provide value by prioritizing beneficiaries for additional assessment. However, more resource-intensive interventions would likely require additional predictors or more targeted thresholds to improve precision.

## Overall Modeling Takeaway

The models demonstrate that **future high-cost risk is partially predictable from beneficiary characteristics, prior utilization, spending, and clinical burden**, particularly when historical information is available.

The results also reveal an important distinction between **risk ranking and high-confidence classification**. XGBoost provides strong discriminative performance and enables detailed feature-level interpretation, while logistic regression provides substantially higher recall at the selected operating point and is therefore better aligned with a broad screening strategy.

More broadly, the modeling results reinforce the central finding from the EDA:

> **High-cost healthcare utilization is driven by a combination of clinical complexity, historical utilization, and spending patterns rather than by utilization volume alone.**

The findings support the potential use of claims-based risk stratification as a first-stage screening tool, while also highlighting the need for richer longitudinal information and more detailed service-intensity features to improve precision and support higher-stakes intervention targeting.


