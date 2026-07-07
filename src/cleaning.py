import pandas as pd
import numpy as np

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
        22: "MA",
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
    df = df.copy()

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


    
    
    
    