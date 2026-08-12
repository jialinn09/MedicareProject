import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

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
        id_col (int): the identifier to merge tables on
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
        esrd_history = history[[id_col, "BENE_ESRD_IND"]].copy()
        esrd_history = esrd_history.rename(columns={"BENE_ESRD_IND": "prior_ESRD"})
        X = X.merge(esrd_history, on=id_col, how="left")
        history.drop(columns=BASELINE_FEATURES, inplace=True)
        history_features = (CHRONIC_COUNT_FEATURES + 
                            DISEASE_FEATURES +
                            UTILIZATION_FEATURES +
                            COST_FEATURES +
                            COVERAGE_FEATURES +
                            RISK_FEATURES)
        history = history[[id_col] + history_features]
        history = history.rename(columns={col:f"2008_{col}" for col in history_features})
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
        # create the flag for prior ESRD exposure
        esrd_history = history.groupby(id_col)["BENE_ESRD_IND"].agg(prior_ESRD="max").reset_index()
        # retrieve the recent historical data
        recent = history[history["year"] == 2009].copy()
        recent_features = UTILIZATION_FEATURES + COST_FEATURES + COVERAGE_FEATURES
        recent = recent[[id_col] + recent_features].copy()
        recent = recent.rename(columns={col: f"2009_{col}" for col in recent_features})

        for features in [disease_history, chronic_history, cumulative, risk_history, recent, esrd_history]:
            X = X.merge(features, on=id_col, how="left")
        y = current[TARGETS].copy()
        X = X.drop(columns=[id_col])

        # lower model performance led to investigations in multicollinearity, which then led to the distinction of which terms to be left cumulatives and which to be 2009 only
        X.drop(columns=["2009_num_carrier_claims",	"2009_num_ip_claims", 
                     "2009_num_op_claims",	"2009_num_pde_claims", 
                     "2009_total_days_medsupplied", "2009_total_carrier_insurance_cost", 
                     "2009_total_carrier_beneficiary_cost", "2009_total_inpatient_beneficiary_cost", 
                     "2009_total_inpatient_insurance_cost_adj", "2009_total_outpatient_beneficiary_cost",	
                     "2009_total_medoop",	"2009_total_raw_drug_cost",	
                     "2009_total_outpatient_insurance_cost_adj", "2009_claim_processing_burden",
                     "2009_PART_A_INSUR_CVRAGE_TOT_MONS",	"2009_PART_B_INSUR_CVRAGE_TOT_MONS",
                     "2009_HMO_INSUR_CVRAGE_TOT_MONS",	"2009_PART_D_INSUR_CVRAGE_TOT_MONS"], inplace=True)
        # part A insurance coverage and part B had high overlaps according to VIF
        X.drop(columns=["cum_num_pde_claims", "cum_PART_B_INSUR_CVRAGE_TOT_MONS", "cum_total_carrier_beneficiary_cost"], inplace=True)
        return X, y

def fix_feature_dtypes(X):
    """
    Convert duration features to numeric days and
    ensure categorical variables have correct dtypes.

    Args:         
        X (pd.DataFrame): feature matrix
    Returns:
        X (pd.DataFrame): feature matrix with dtype properly fixed

    """
    duration_cols = [col for col in X.columns if "claim_duration" in col]
    for col in duration_cols:
        X[col] = pd.to_timedelta(X[col]).dt.days
    X["BENE_COUNTY_CD"] = X["BENE_COUNTY_CD"].astype(str)
    return X

def run_logistic_regression_experiment(X, y):
    """
    Train and evaluate logistic regression for a single target.

    Args:
        X (pd.DataFrame): feature matrix
        y (pd.DataFrame): true labels

    Returns:
        X2 (pd.DataFrame): evaluation metrics for both prediction targets
    """
    # Transform features
    categorical_features = ["BENE_COUNTY_CD", "SEX", "RACE", "STATE"]
    numeric_features = [col for col in X.columns if col not in categorical_features]
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"),
         categorical_features)])
    
    # Create the model
    lr_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42))
    ])

    results = []
    for target in ["High_Payer_Cost", "High_OOP_Burden"]:
        y_target = y[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_target,
            test_size=0.2,
            stratify=y_target,
            random_state=42
        )
        lr_pipeline.fit(X_train, y_train)
        metrics = evaluate_model(lr_pipeline, X_test, y_test)
        results.append({"target": target, **metrics})

    return pd.DataFrame(results)
    
def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance and output classification report and confusion matrix

    Args:
        model: fitted sklearn-compatible model
        X_test (pd.DataFrame): test feature matrix
        y_test (pd.Series): true labels for the test set
    Returns:
        Dictionary (dict): containing summary evaluation metrics, including ROC-AUC. 
    """
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("=" * 60)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    print(
        f"ROC-AUC: "
        f"{roc_auc_score(y_test, y_prob):.4f}"
    )

    print("\nClassification Report")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred))

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }