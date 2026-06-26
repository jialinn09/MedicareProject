import zipfile
import pandas as pd

def load_csv_from_zip(zip_path: str):
    """
    Load a CSV file from a zip archive.

    Args:
        zip_path (str): Path to zip file containing exactly one CSV.

    Returns:
        pd.DataFrame
    """
    with zipfile.ZipFile(zip_path, 'r') as z:
        file_list = z.namelist()
        csv_files = [f for f in file_list if f.endswith(".csv")]
        if len(csv_files) != 1:
            raise ValueError(f"Expected 1 CSV, found {len(csv_files)} in {zip_path}")
        csv_file = csv_files[0]
        with z.open(csv_file) as f:
            return pd.read_csv(f, low_memory=False)

def load_claim_from_path(path):
    """
    Loads claim from the given path.

    Args:
        path: can be both dictionary type and list type of single/multiple paths.

    Returns:
        pd.DataFrame
    """
    dfs = []

    # Case 1: Dictionary path
    if isinstance(path, dict):
        for year, p in path.items():
            df = load_csv_from_zip(p)
            df["year"] = year
            dfs.append(df)
            
    # Case 2: List path
    elif isinstance(path, list):
        for p in path:
            df = load_csv_from_zip(p)
            dfs.append(df)
            
    else:
        raise ValueError(f"Unsupported path type: {type(path)}")
        
    return pd.concat(dfs, ignore_index=True)
            