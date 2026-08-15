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
