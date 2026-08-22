# Predicting High-Cost Medicare Patients

**[→ Interactive Tableau Dashboard](https://public.tableau.com/app/profile/jialin.jiang4317/viz/MedicareAnalysis_17861454891190/MedicarePopulationOverview)** · **[→ Full Analysis Write-Up](./ANALYSIS.md)** · **[→ Methodology](./METHODS.md)**

**Objective:** Identify future high-cost Medicare beneficiaries and characterize patterns associated with healthcare utilization and financial burden to inform payer intervention strategies.  
**Data:** CMS DE-SynPUF (2008–2010) Sample 9 | 332,694 patient-year records | 110,898 unique beneficiaries  
**Tools:** Python (3.13.5), Scikit-Learn (1.6.1), XGBoost (3.1.2), SHAP (0.52.0), Tableau Public

---

## Executive Summary: Why This Matters

*   **For Health Analytics:** Healthcare utilization *volume* does not directly translate to *cost*. Outpatient care intensity and composition matter far more than simple encounter counts. Additionally, ESRD is a major clinical indicator for disproportionate medical costs.
*   **For Payer Strategy:** High-cost beneficiaries have distinct risk phenotypes. The "high payer + high OOP (out-of-pocket)" subpopulation drives inpatient/outpatient costs. Meanwhile, the "high OOP/low payer burden" group is uniquely driven by prescription drug cost-sharing.
*   **For Predictive Modeling:** Historical utilization substantially improves prediction of future high-cost beneficiaries, with models achieving up to **0.90 ROC-AUC** on held-out test data. Risk stratification into quintiles captured **51.1% of future high-cost beneficiaries** (51.1% recall) by targeting the top 20% predicted-risk population.

---
## Key Findings

**1. Demographic Representation & Limitations** 

<img width="786" height="330" alt="Screenshot 2026-08-15 at 1 55 11 PM" src="https://github.com/user-attachments/assets/c4a9fa06-3c88-4df3-a182-f2670ecfaf83" />

*The dataset is disproportionately represented by White beneficiaries and women (55.57% of records); White beneficiaries comprise ~83% of both female and male records. Given the sample's demographic skew, we did not have sufficient subgroup sample size to validate whether risk-phenotype patterns generalize across race; this is a priority for future work with a more representative dataset.*

**2. Distinct Cost-Burden Phenotypes**  
| Profile | Interpretation | Example subgroups |
|---|---|---|
| 🟧 High OOP burden, lower payer spending | High OOP prevalence, lower payer prevalence | Hispanic females 75–84; Hispanic males 85–94 |
| 🟨 High payer spending, lower OOP burden | High payer prevalence, lower OOP prevalence | Black males 95+ |
| 🟥 High prevalence of both | High prevalence of both | Females 85–94 and <65; non-Hispanic males 85–94; non-Black males 95+ |
| 🟩 Low prevalence of both | Low prevalence of both | Ages 65–74 across demographic groups |

<img width="764" height="540" alt="Screenshot 2026-08-15 at 9 37 55 PM" src="https://github.com/user-attachments/assets/28fb81a0-9d73-4067-8596-61a53dc86d9e" />

*Distinct service-type count alone does not capture cost intensity: some of the highest-cost subgroups, including Black males 95+ and non-Black males 95+, show relatively low average service-type utilization despite high financial burden.*

**3. The Outpatient Utilization Paradox**  

<img width="590" height="546" alt="OP util" src="https://github.com/user-attachments/assets/4fce10ba-9a30-4e4e-a4cd-2ad1843289d2" />

*The high-OOP/lower-payer and low-burden-both groups show higher outpatient utilization than the high-payer/lower-OOP group despite having lower payer costs. This suggests outpatient cost depends more on encounter intensity and procedure composition than on raw visit volumes.*

**4. Disease Burden Aligns with OOP Burden**  

<img width="478" height="546" alt="Screenshot 2026-08-16 at 11 39 10 AM" src="https://github.com/user-attachments/assets/d38e070e-6733-4811-89c8-8764bc61b987" />

*Across six chronic conditions (diabetes, cancer, CKD, COPD, ESRD, heart failure), a consistent ordering emerges: High-burden-both → High OOP/lower payer → Low burden-both → High payer/lower OOP. Clinical disease burden appears more closely associated with patient OOP burden than with payer-only cost burden. This suggests high payer spending and high OOP burden may represent different risk phenotypes.*

**[→ Explore the full interactive Medicare Analysis Dashboard on Tableau Public](https://public.tableau.com/app/profile/jialin.jiang4317/viz/MedicareAnalysis_17861454891190/MedicarePopulationOverview)**

---

## Repository Structure

*   `code_mapping/`: Reference for mapping ICD-9 codes into distinct disease categories.
*   `src/`: Modular, reusable Python functions for data cleaning, feature engineering, modeling, and SHAP analysis.
*   `results/`: Generated outputs, including SHAP summary/dependence plots, model evaluation metrics, feature importance, and risk stratification tables (patient-level and quintile-level).
*   `notebooks/`: All Jupyter notebooks to run within this project.
    *   `0_data_loading.ipynb`: Source data ingestion.
    *   `1_data_cleaning.ipynb`: Auditing, validation, and feature engineering.
    *   `2_eda_and_feature_engineering.ipynb`: Population, utilization, and disease burden analysis.
    *   `3_modeling.ipynb`: Logistic Regression, Random Forest, and XGBoost training.
    *   `4_analyzing.ipynb`: SHAP feature importance and risk stratification.
*   `requirements.txt`: Contains the dependencies for this project.
*   `ANALYSIS.md`: Full, detailed write-up of findings from the project.
*   `METHODS.md`: Deep dive into methodology, design choices, and validation strategies.

---

## Dataset & Limitations

*   **Synthetic Data:** Uses CMS DE-SynPUF; patterns are structurally accurate but do not represent real individuals.
*   **Historical Context:** Covers 2008–2010; healthcare spending dynamics have evolved since.
*   **Correlation ≠ Causation:** Predictive associations identify risk markers, not causal drivers.
*   **Relative Thresholds:** High-cost thresholds are relative to the annual sample distribution, not universal clinical benchmarks.

*View the Full Analysis in [ANALYSIS.md](ANALYSIS.md)*

---
## Reproducibility & Setup

This project is fully reproducible and engineered for frictionless execution. The core logic is modularized in the `src/` directory, and the pipeline automatically generates the `results/` directory and saves all outputs in the correct locations—no manual folder setup required. Outputs are deterministic (random seed = 42).

> **Note:** Due to file size and data governance best practices, the raw CMS DE-SynPUF dataset is not committed to this repository. Follow the steps below to set up the data locally.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jialinn09/MedicareProject.git
   cd MedicareProject

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   
3. **Acquire the dataset**:
- Download all 8 data files from the CMS 2008-2010 Data Entrepreneurs’ Synthetic Public Use File (DE-SynPUF) [Sample 9 data records](https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files/cms-2008-2010-data-entrepreneurs-synthetic-public-use-file-de-synpuf/de10-sample-9?spm=a2ty_o01.29997173.0.0.7ebc55fbr53eDo).
- Create a data/raw/ directory in the root of this project and place the downloaded files inside:
  - Beneficiary & Carrier Data: Keep as .zip files.
  - Inpatient, Outpatient, & PDE: Ensure these are .csv files.
    
4. **Run the pipeline**:
- Execute the notebooks in numerical order (0_ through 4_).
- Once complete, the pipeline will automatically generate a results/ directory containing all SHAP plots, model evaluation metrics, and risk stratification tables referenced in this README and the ANALYSIS.md report.

**Performance Note**: The Carrier Claims dataset contains ~34M+ records per year. The data loading and merging step for Carrier data will take several minutes to process. This is expected behavior as the pipeline absorbs this complexity to output a highly optimized, feature-engineered dataset.
