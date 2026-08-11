import pandas as pd
import numpy as np

BASELINE_FEATURES = [
    "BENE_ESRD_IND",
    "BENE_COUNTY_CD",
    "SEX",
    "RACE",
    "STATE",
    "age",
]

DISEASE_FEATURES = [
    "has_Additional endocrine/metabolic burden",
    "has_COPD",
    "has_Administrative_screening_aftercare",
    "has_Cancer",
    "has_Injury/Frailty Burden",
    "has_Additional cardiovascular burden",
    "has_Additional mental health burden",
    "has_Additional Musculoskeletal Burden",
    "has_Infectious disease burden",
    "has_Stroke/TIA",
    "has_Additional neurologic burden",
    "has_Diabetes",
    "has_Additional renal burden",
    "has_Genitourinary (non-renal) burden",
    "has_Sensory Burden",
    "has_Additional Hematologic burden",
    "has_Rheumatoid Arthritis/Osteoarthritis",
    "has_Depression",
    "has_Heart Failure",
    "has_CKD",
    "has_Additional GI/Hepatic burden",
]

CHRONIC_COUNT_FEATURES = [
    "max_carrier_chronic_code_counts",
    "max_ip_chronic_code_counts",
    "max_op_chronic_code_counts",
]

UTILIZATION_FEATURES = [
    "num_carrier_claims",
    "avg_carrier_claim_duration",
    "num_ip_claims",
    "avg_ip_claim_duration",
    "num_op_claims",
    "avg_op_claim_duration",
    "num_pde_claims",
    "total_days_medsupplied",
    "avg_days_supplied_per_pdeclaim",
]

COST_FEATURES = [
    "avg_carrier_rejected_amt",
    "avg_carrier_reimb_allowed_diff",
    "avg_carrier_insurance_cost",
    "total_carrier_insurance_cost",
    "avg_carrier_beneficiary_cost",
    "total_carrier_beneficiary_cost",
    "avg_inpatient_beneficiary_cost",
    "total_inpatient_beneficiary_cost",
    "avg_inpatient_insurance_cost",
    "total_inpatient_insurance_cost_adj",
    "avg_outpatient_insurance_cost",
    "avg_outpatient_beneficiary_cost",
    "total_outpatient_beneficiary_cost",
    "total_medoop",
    "total_raw_drug_cost",
    "total_outpatient_insurance_cost_adj",
    "claim_processing_burden",
]

COVERAGE_FEATURES = [
    "PART_A_INSUR_CVRAGE_TOT_MONS",
    "PART_B_INSUR_CVRAGE_TOT_MONS",
    "HMO_INSUR_CVRAGE_TOT_MONS",
    "PART_D_INSUR_CVRAGE_TOT_MONS",
]

RISK_FEATURES = [
    "High_Payer_Cost",
    "High_OOP_Burden",
]

TARGETS = [
    "High_Payer_Cost",
    "High_OOP_Burden",
]

def build_features(df, year, id_col="DESYNPUF_ID"):
    """
    Build year-specific prediction features while preventing
    future information from entering the feature set.

    Args:
        df (pd.DataFrame): the reference data table
        year (int): the year to check for
            2008:
                Baseline characteristics only.

            2009:
                Baseline characteristics + 2008 history.

            2010:
                Baseline characteristics + 2008-2009 history.
    Returns:
        X (pd.DataFrame): the feature matrix for the given year.
        y (pd.DataFrame): the two prediction targets: "High_Payer_Cost" and "High_OOP_Burden".
    """

    if year not in [2008, 2009, 2010]:
        raise ValueError("year must be 2008, 2009, or 2010")
        
    # subset the dataframe to include just those of the given year 
    current = df[df["year"] == year].copy()

    # Keep beneficiary ID for merging, but not using it as a feature
    X = current[[id_col] + BASELINE_FEATURES].copy()

    # For 2008, we'll just have demographic as the baseline
    if year == 2008:
        y = current[TARGETS].copy()
        X = X.drop(columns=[id_col])
        return X, y

    # For 2009, we need to merge 2008 historical records with the current 2009 ID + baseline
    elif year == 2009:
        history = df[df["year"] == 2008].copy()
        history_features = (CHRONIC_COUNT_FEATURES + 
                            DISEASE_FEATURES +
                            UTILIZATION_FEATURES +
                            COST_FEATURES +
                            COVERAGE_FEATURES +
                            RISK_FEATURES)
        history = history.rename(columns={col:f"2008_{col}" for col in history_features)
        X = X.merge(history, on=id_col, how="left")
        y = current[TARGETS].copy()
        X = X.drop(columns=[id_col])
        return X, y

    # For 2010, we need to have cumulative records
    else:
        history = df[df["year"].isin([2008, 2009])].copy()
        # create the prior disease flag variable, returning 1 if this disease appeared in 2008/9
        disease_history = history.groupby(id_col)[DISEASE_FEATURES].max().reset_index()
        disease_history = disease_history.rename(columns={col: f"prior_{col}" for col in
                                                          DISEASE_FEATURES})
        # create the cumulative max chronic code count across 2008/9
        chronic_history = history.groupby(id_col)[CHRONIC_COUNT_FEATURES].max().reset_index()
        chronic_history = chronic_history.rename(columns={col: f"prior_{col}" for col in 
                                                          CHRONIC_COUNT_FEATURES})
        # create the cumulative variables for prior utilization, cost, and coverage
        SUM_FEATURES = [
        "num_carrier_claims",
        "num_ip_claims",
        "num_op_claims",
        "num_pde_claims",
        "total_days_medsupplied",
        "total_carrier_insurance_cost",
        "total_carrier_beneficiary_cost",
        "total_inpatient_beneficiary_cost",
        "total_inpatient_insurance_cost_adj",
        "total_outpatient_beneficiary_cost",
        "total_medoop",
        "total_raw_drug_cost",
        "total_outpatient_insurance_cost_adj",
        "claim_processing_burden"]
        
        SUM_FEATURES += COVERAGE_FEATURES
        cumulative = history.groupby(id_col)[SUM_FEATURES].sum().reset_index()
        cumulative = cumulative.rename(columns={col: f"cum_{col}" for col in SUM_FEATURES})
        
        # create the prior risk flag count variables
        risk_history = history.groupby(id_col)[RISK_FEATURES].sum().reset_index()
        risk_history = risk_history.rename(columns={"High_Payer_Cost":
                                                    "prior_High_Payer_Cost_counts",
                                                    "High_OOP_Burden":
                                                    "prior_High_OOP_Burden_counts"
                                                   })
        # retrieve the recent historical data
        recent = history[history["year"] == 2009].copy()
        recent_features = UTILIZATION_FEATURES + COST_FEATURES + COVERAGE_FEATURES
        recent = recent[[id_col] + recent_features].copy()
        recent = recent.rename(columns={col: f"2009_{col}" for col in recent_features})

        for features in [disease_history, chronic_history, cumulative, risk_history, recent]:
            X = X.merge(features, on=id_col, how="left")
        y = current[TARGETS].copy()
        X = X.drop(columns=[id_col])
        return X, y