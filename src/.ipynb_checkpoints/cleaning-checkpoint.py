import pandas as pd
import numpy as np
from icdmappings import Mapper

def print_table(df):
    """
    print out the dataframe for clear tabular data preview

    Args:
        df (pd.DataFrame): the data table to print

    Returns:
        None type
    """
    with pd.option_context(
        "display.max_columns", None,
        "display.width", None,
        "display.max_colwidth", None
    ):
        print(df)
        
def audit_df(df: pd.DataFrame, name: str, id_col: str):
    """
    Performs an audit check on the dataframe and returns results on missingness and the rows with duplications -- helping with data understanding and cleaning.

    Args:
        df (pd.DataFrame): the data table to perform audit checks on.
        name (str): the name of this data table.
        id_col (str): the unique identifier of each row.

    Returns:
        dictionary list
    """
    print(name)

    # 1. find the shape of the data
    print(f"\nRows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    # 2. check for the number of unique ids
    print(f"Unique {id_col}: {df[id_col].nunique()}")

    # 3. check for duplicates
    duplicated_counts = df.duplicated().sum()
    duplicated_rows = df[df.duplicated()]
    print(f"Duplicate rows: {duplicated_counts}")

    # 4. check for missingness
    missing_summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_count": df.isna().sum(),
        "missing_percentage": round(df.isna().mean() * 100, 2)
    })
    missing_summary = missing_summary.sort_values(
        "missing_percentage", ascending=False)
    print("\n Top 10 columns with highest missingness:")
    print(
        missing_summary.head(10)[["missing_count", "missing_percentage"]])
    
    # 5. show the first 5 rows
    print("\nFirst 5 rows:")
    print_table(df.head(5))
    
    # 6. return audit results
    return {
        "missingness": missing_summary,
        "duplicated_rows": duplicated_rows
    }


def mapping_beneficiary_demographics(df: pd.DataFrame):
    """
    Map categorical variables according to code book to be more informative and
    adjust dtype for death date and birth date as well for the beneficiary

    Args:
        df (pd.DataFrame): the data table to perform definition mappings to.

    Returns:
        df (pd.DataFrame): the properly mapped dataframe.
        
    """
    df = df.copy()
    
    # Sex mapping
    sex_map = {1: "Male", 2: "Female"}
    df["SEX"] = df["BENE_SEX_IDENT_CD"].map(sex_map)
    
    # Race mapping. Others belong to the category of unknown, Asian, Hispanic, and North American Native
    race_map = {
        1: "White",
        2: "Black",
        3: "Others",
        5: "Hispanic"
    }
    df["RACE"] = df["BENE_RACE_CD"].map(race_map)
    
    state_map = {
        1: "AL", 
        2: "AK", 
        3: "AZ", 
        4: "AR", 
        5: "CA", 
        6: "CO", 
        7: "CT", 
        8: "DE", 
        9: "DC", 
        10: "FL", 
        11: "GA",
        12: "HI", 
        13: "ID", 
        14: "IL", 
        15: "IN", 
        16: "IA", 
        17: "KS", 
        18: "KY", 
        19: "LA", 
        20: "ME", 
        21: "MD", 
        23: "MI", 
        24: "MN", 
        25: "MS", 
        26: "MO", 
        27: "MT", 
        28: "NE", 
        29: "NV", 
        30: "NH", 
        31: "NJ", 
        32: "NM", 
        33: "NY",
        34: "NC", 
        35: "ND", 
        36: "OH", 
        37: "OK", 
        38: "OR", 
        39: "PA", 
        41: "RI", 
        42: "SC", 
        43: "SD", 
        44: "TN", 
        45: "TX",
        46: "UT", 
        47: "VT", 
        49: "VA", 
        50: "WA", 
        51: "WV", 
        52: "WI", 
        53: "WY", 
        54: "Others"
    }
    
    # others belong to the category of Puerto Rico, Virgin Islands, Africa, Asia, or California, Canada & Islands, Central America, West Indies, Europe, Mexico, Oceania, Philippines, South America, American Samoa, Guam, Sapian, Northern Marianas, Texas, Guam, and unknown
    df["STATE"] = df["SP_STATE_CODE"].map(state_map)
    
    # change dtypes properly
    df = df.astype({
        "BENE_SEX_IDENT_CD": "category",
        "SEX": "object",
        "BENE_RACE_CD": "category",
        "RACE": "object",
        "SP_STATE_CODE": "category",
        "STATE": "object",
        "BENE_COUNTY_CD": "category"
    })
    
    # adjust the date formatting for birth/death dates
    df["BENE_BIRTH_DT"] = pd.to_datetime(
        df["BENE_BIRTH_DT"].astype(str),
        format="%Y%m%d",
        errors="coerce"
    )
    df["BENE_DEATH_DT"] = pd.to_datetime(
        df["BENE_DEATH_DT"].astype(str),
        format="%Y%m%d",
        errors="coerce"
    )
    return df

def recoding_beneficiary_flags(df: pd.DataFrame):
    """
    Recode disease flags to be 0/1 binary outputs for easier downstream operations.

    Args:
        df (pd.DataFrame): the data table to perform recoding checks on.

    Returns:
        df (pd.DataFrame): the properly recoded dataframe.
        
    """
    df = df.copy()

    flag_cols = [
        "BENE_ESRD_IND", 
        "SP_ALZHDMTA", 
        "SP_CHF", 
        "SP_CHRNKIDN", 
        "SP_CNCR", 
        "SP_DEPRESSN", 
        "SP_DIABETES", 
        "SP_ISCHMCHT", 
        "SP_OSTEOPRS", 
        "SP_RA_OA", 
        "SP_STRKETIA"
    ]
    for col in flag_cols:
        df[col] = (df[col].astype(str).str.strip().str.upper().replace({
            "Y": 1, 
            "N": 0,
            "1": 1,
            "2": 0,
              2: 0
        }))
        pd.set_option('future.no_silent_downcasting', True)
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        
    return df




def checking_claims_logic(df: pd.DataFrame):
    """
    Clean carrier line-level payments using processing indicators, then check reconciliation between reimbursement and allowed charge.

    Includes:
    - Rejected (B, C, D, N): payment zeroed; calculate the rejected $ amount and % of total payment 
    - Non-payable (I, P, Z): payment zeroed, no further tracking
    - MSP-related codes: flagged; excluded from strict reconciliation check since primary payer amounts aren't bounded by Medicare's allowed charge
    - Review-pending (O): flagged via needs_review, left unmodified
    - M (duplicate): NOT specially handled. Inspection of sample rows showed inconsistent payment amounts between M-flagged lines and their presumed duplicate pair (e.g., $20 vs $40 for same HCPCS code), so no reliable zero-out logic could be justified from the data. Left as-is; documented as a known limitation.

    Args:
        df (pd.DataFrame): claim-level carrier data with LINE_* amount
                           columns and LINE_PRCSG_IND_CD_1 through _13.
    Returns:
        df (pd.DataFrame): dataframe with aggregated stakeholder 
                           payments, rejection diagnostics, 
                           MSP/review flags, and a reconciliation
                           check column.
    """

    ind_cols = [f"LINE_PRCSG_IND_CD_{i}" for i in range(1, 14)]

    rejected_codes = {"B", "C", "D", "N"}
    nonpayable_codes = {"I", "P", "Z"}
    msp_codes = {
        "Q", "S", "T", "U", "V", "X", "Y",
        "00", "12", "13", "14", "15", "16", "17", "18", "21", "22",
        "25", "26", "!", "@", "#", "$", "*", "(", ")", "+", 
        "<", ">", "%", "&",
    }
    review_codes = {"O"}

    amount_prefixes = [
        "LINE_NCH_PMT_AMT",
        "LINE_COINSRNC_AMT",
        "LINE_BENE_PTB_DDCTBL_AMT",
        "LINE_BENE_PRMRY_PYR_PD_AMT",
    ]

    # track the rejected payment amounts per line
    rejected_pmt_cols = []
    for i in range(1, 14):
        ind_col = f"LINE_PRCSG_IND_CD_{i}"
        pmt_col = f"LINE_NCH_PMT_AMT_{i}"
        temp_col = f"_rejected_pmt_{i}"
        is_rejected = df[ind_col].isin(rejected_codes)
        df[temp_col] = np.where(is_rejected, df[pmt_col], 0)
        rejected_pmt_cols.append(temp_col)

    df["CARRIER_REJECTED_AMT"] = df[rejected_pmt_cols].sum(axis=1)
    df = df.drop(columns=rejected_pmt_cols)

    # zero out rejected and non-payable line values
    zero_out_codes = rejected_codes | nonpayable_codes
    for i in range(1, 14):
        ind_col = f"LINE_PRCSG_IND_CD_{i}"
        should_zero = df[ind_col].isin(zero_out_codes)
        for prefix in amount_prefixes:
            amt_col = f"{prefix}_{i}"
            df[amt_col] = np.where(should_zero, 0, df[amt_col])

    # aggregate stakeholder payments
    prefix_map = {
        "LINE_NCH_PMT_AMT": "CARRIER_NCHREIMB",
        "LINE_COINSRNC_AMT": "CARRIER_COINS",
        "LINE_BENE_PTB_DDCTBL_AMT": "CARRIER_DDCTBL",
        "LINE_BENE_PRMRY_PYR_PD_AMT": "CARRIER_PPPYMT",
        "LINE_ALOWD_CHRG_AMT": "CARRIER_ALLOWED",
    }
    for prefix, label in prefix_map.items():
        cols = [c for c in df.columns if c.startswith(prefix)]
        df[label] = df[cols].sum(axis=1)

    df["CARRIER_BENERES"] = df["CARRIER_DDCTBL"] + df["CARRIER_COINS"]
    df["CARRIER_TOTAL_REIMB"] = (
                                df["CARRIER_NCHREIMB"] +
                                df ["CARRIER_BENERES"] + 
                                df["CARRIER_PPPYMT"]
    )

    # calculate total medical cost before rejecting and the reject %
    df["CARRIER_TOTAL_PMT_PRE_REJECT"] = df["CARRIER_TOTAL_REIMB"] + df["CARRIER_REJECTED_AMT"]
    df["pct_rejected"] = (
                          df["CARRIER_REJECTED_AMT"] /
        df["CARRIER_TOTAL_PMT_PRE_REJECT"].replace(0, np.nan)
    )

    # set flags
    df["has_msp_line"] = df[ind_cols].isin(msp_codes).any(axis=1)
    df["needs_review"] = df[ind_cols].isin(review_codes).any(axis=1)

    # reconciliation check
    base_check = df["CARRIER_TOTAL_REIMB"] <= df["CARRIER_ALLOWED"] + 0.01
    df["reimb_valid"] = np.where(df["has_msp_line"], True, base_check)

    return df

def icd_mappings(df: pd.DataFrame, count: int, special_col: str=None):
    """
    Map ICD-9 diagnostic codes into CCS and CCI indicators. 

    Args:
        df (pd.DataFrame): the data table to perform icd mappings on.
        count (int): number of ICD diagnosis columns to map + 1 to buffer.
        special_col (str, optional): additional ICD-9 column to map if it does not follow the standard naming convention (e.g., "ADMTNG_ICD9_DGNS_CD").


    Returns:
        df (pd.DataFrame): the data table with clinically meaningful disease categorizations.
    """
    mapper = Mapper()
    diag_cols = [f"ICD9_DGNS_CD_{i}" for i in range(1,count)]
    if special_col is not None:
        diag_cols.append(special_col)
   
    # Step 1: get all unique ICD codes without melting which explodes row counts
    unique_codes = pd.Series(pd.unique(df[diag_cols].values.ravel())).dropna()

    # Step 2: set up the dictionary
    cci_lookup = {}
    ccs_lookup = {}
    for code in unique_codes:
        try:
        # chronic condition flag
            cci_lookup[code] = mapper.map(code, source='icd9', target='cci')
        except Exception:
            cci_lookup[code] = None
        try:
        # form clinical categories
            ccs_lookup[code] = mapper.map(code, source='icd9', target='ccs')
        except Exception:
            ccs_lookup[code] = None

    # Step 3: map column-by-column
    for i, diag_col in enumerate(diag_cols, start=1):
        df[f"CCI_{i}"] = df[diag_col].map(cci_lookup).astype("float32")
        df[f"CCS_{i}"] = df[diag_col].map(ccs_lookup).astype("float32")

    return df


def disease_burden_mappings(df: pd.DataFrame):
    """
    Map CCS codes into disease burden buckets. 

    Args:
        df (pd.DataFrame): the data table to perform categorical mappings on.

    Returns:
        df (pd.DataFrame): the data table with clinically meaningful disease categorizations.
    """
    disease_burden_map = {"Alzheimer's/Dementia": {653},
                          "Heart Failure": {108},
                          "CKD": {158},
                          "Cancer": set(range(11, 46)),
                          "COPD": {127, 131},
                          "Depression": {657},
                          "Diabetes": {49, 50},
                          "Ischemic Heart Disease": {100, 101},
                          "Osteoporosis": {206},
                          "Rheumatoid Arthritis/Osteoarthritis": {202, 203},
                          "Stroke/TIA": set(range(109, 114)),
                          "Infectious disease burden": set(range(1, 11)) | 
                          set(range(122, 127)) | {135, 197, 201},
                          "Additional cardiovascular burden": set(range(96, 108)) |
                          set(range(114, 122)) | {98, 99},
                          "Additional respiratory burden": set(range(128-135)) |{56},
                          "Additional neurologic burden": set(range(76, 86)) | 
                          set(range(79, 84)) | {95},
                          "Additional mental health burden": set(range(650, 653)) | 
                          set(range(645, 657)) | set(range(658, 664)) | {670},
                          "Additional endocrine/metabolic burden": {48} | set(range(51, 59)),
                          "Additional Hematologic burden": set(range(59, 65)),
                          "Additional GI/Hepatic burden": set(range(136, 156)) | 
                          set(range(149, 152)),
                          "Additional renal burden": set(range(156, 158)) | 
                          set(range(159, 164)), 
                          "Additional Musculoskeletal Burden": set(range(204, 213)) | {54},
                          "Sensory Burden": set(range(86, 95)),
                          "Injury/Frailty Burden": set(range(225, 245)) | 
                          set(range(259,261)) | set(range(2601, 2622)) | {252, 245, 248, 249},
                          "Genitourinary (non-renal) burden": set(range(164, 176)),
                          "Administrative_screening_aftercare": {254, 255, 256, 257, 258}
                         }
    disease_category_lookup = {code:disease for disease, codes in 
                               disease_burden_map.items() for code in codes}
    diag_ccs_cols = [c for c in df.columns if c.startswith("CCS_")]

    for col in diag_ccs_cols:
        df[col.replace("CCS", "DISEASE_CATEGORY")] = df[col].map(disease_category_lookup)
        
    return df


def disease_mapping_cleaned(df: pd.DataFrame):
    """
    Clean up the disease_burden_mapping artifacts to one-hot-encode the diseases and also include a count of claim lines and the amount of chronic codes per claim.. 

    Args:
        df (pd.DataFrame): the data table to perform categorical mappings on.

    Returns:
        df (pd.DataFrame): the data table with clinically meaningful disease categorizations.
    """
    # dropping ICD9 and CCS columns now since disease buckets are informative on its own
    prefix_columns = [c for c in df.columns if c.startswith(("ICD9_DGNS_CD", "CCS"))]
    df_cleaned = df.drop(columns=prefix_columns)
    
    # one-hot-encode the diseases
    disease_cols = [c for c in df_cleaned.columns if c.startswith("DISEASE_CATEGORY_")]

    all_diseases = df_cleaned[disease_cols].stack().dropna().unique()
    for disease in all_diseases:
        df_cleaned[f"has_{disease}"] = df_cleaned[disease_cols].eq(disease).any(axis=1).astype(int)
        
    # compute the claim line counts and amount of chronic codes per claim
    claim_line_cols = [c for c in df_cleaned.columns if c.startswith("CCI_")]
    disease_columns = [c for c in df_cleaned.columns if c.startswith("has_")]
    df_cleaned["num_claim_lines"] = df_cleaned[disease_columns].sum(axis=1)
    df_cleaned["num_chronic_codes"] = df_cleaned[claim_line_cols].sum(axis=1)
    df_cleaned.drop(columns = claim_line_cols, inplace=True)
    
    # drop the disease categories
    df_cleaned.drop(columns = disease_cols, inplace=True)

    return df_cleaned
    

def carrier_table_aggregation(df: pd.DataFrame):
    """
    Clean and aggregate carriers into patient-year granularity for downstream work. 

    Args:
        df (pd.DataFrame): the data table to perform cleaning and aggregations on.

    Returns:
        df (pd.DataFrame): the data table with correct granularities.
    """
    # Step 1: Cleaning
    disease_cols = [c for c in df.columns if c.startswith("has_")]
    df.drop(columns=["CLM_FROM_DT", "CLM_THRU_DT"], inplace=True)

    # Step 2: Aggregating
    agg_dict = {
        "CLM_ID": "count",
        "claim_duration": ["mean", "max", "sum"],
        "CARRIER_REJECTED_AMT": ["mean", "max", "sum"],
        "pct_rejected": ["mean", "max"],
        "overage": ["mean", "min", "max", "sum"],
        "num_chronic_codes":["max"]
    }
    for col in disease_cols:
        agg_dict[col] = "max"
    df_cleaned = df.groupby(["DESYNPUF_ID", "claim_year"]).agg(agg_dict).reset_index()
    df_cleaned.columns = ["_".join(col).strip("_") if isinstance(col, tuple) else col for col in df_cleaned.columns]

    # Step 3: Renaming columns
    df_cleaned = df_cleaned.rename(columns={
        "CLM_ID_count": "num_claim_lines",
        "claim_duration_mean": "avg_claim_duration",
        "claim_duration_max": "max_claim_duration",
        "claim_duration_sum": "total_claim_duration",
        "CARRIER_REJECTED_AMT_mean": "avg_carrier_rejected_amt",
        "CARRIER_REJECTED_AMT_max": "max_carrier_rejected_amt",
        "CARRIER_REJECTED_AMT_sum": "total_carrier_rejected_amt",
        "pct_rejected_mean": "avg_carrier_claim_rejected_rate (%)",
        "pct_rejected_max": "max_carrier_claim_rejected_rate (%)",
        "overage_mean": "avg_reimb_allowed_diff",
        "overage_min": "min_reimb_allowed_diff",
        "overage_max": "max_reimb_allowed_diff",
        "overage_sum": "total_reimb_allowed_diff",
        "num_chronic_codes_max": "max_chronic_code_counts"
    })
    df_cleaned.columns = [col.replace("_max", "") if col.startswith("has_") else col for col in df_cleaned.columns]
    df_cleaned["avg_carrier_claim_rejected_rate (%)"] *= 100
    df_cleaned["max_carrier_claim_rejected_rate (%)"] *= 100
    
    return df_cleaned

def reconcile_disease_flags(dfs, id_cols=["DESYNPUF_ID", "claim_year"]):
    """
    Merge multiple datasets and reconcile overlapping disease flags.
    If any source has a disease/burden flag = 1, final flag = 1.

    Args:
        dfs (list): list of aggregated patient-year datasets
        id_cols (list): merge keys

    Returns:
        pd.DataFrame: merged dataset with reconciled disease flags
    """

    merged = dfs[0].copy()

    for i, df in enumerate(dfs[1:], start=1):
        merged = merged.merge(
            df,
            on=id_cols,
            how="outer",
            suffixes=("", f"_source{i}")
        )

    # identify disease columns
    disease_cols = [
        col for col in merged.columns
        if col.startswith("has_")
    ]

    # reconcile duplicated disease columns
    base_names = set(
        col.split("_source")[0]
        for col in disease_cols
    )

    for disease in base_names:
        related_cols = [
            col for col in disease_cols
            if col == disease or col.startswith(disease + "_source")
        ]

        if len(related_cols) > 1:
            merged[disease] = merged[related_cols].max(axis=1)

            merged.drop(
                columns=[c for c in related_cols if c != disease],
                inplace=True
            )

    return merged




    


    
    
    
    