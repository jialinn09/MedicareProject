# Predicting High-Cost Medicare Patients

## Overview
Built machine learning models to predict which Medicare beneficiaries will have high healthcare costs, enabling early intervention and cost reduction.

## Data
- CMS DE-SynPUF synthetic Medicare claims (2008-2010)
- 115K beneficiaries, ~340K patient-years after excluding 2008

## Approach
- **Feature Engineering:** Historical utilization patterns, chronic conditions, lagged cost averages
- **Models:** Logistic Regression, Random Forest, XGBoost
- **Key Challenge:** Preventing data leakage from current-year costs

## Results
- **Best Model:** XGBoost (AUC = 0.874)
- **Performance:** Identifies 75% of high-cost patients while flagging only 22% of population
- **Top Predictor:** Chronic kidney disease (36% feature importance)


## Key Learnings
Successfully prevented data leakage, optimized decision threshold for business use case, and compared multiple model architectures.