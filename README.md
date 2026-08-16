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



