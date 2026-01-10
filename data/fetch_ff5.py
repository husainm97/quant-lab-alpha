# data/fetch_ff5.py 

import pandas as pd
import zipfile
import io
import requests

def fetch_ff5_monthly():
    """
    Fetches Fama-French 5-Factor monthly data for Developedmarkets,
    parses only the monthly rows, converts percentages to decimals,
    and returns a clean DataFrame with datetime index.
    
    Returns:
    --------
    pd.DataFrame
        Monthly factor returns with columns: Mkt-RF, SMB, HML, RMW, CMA, RF
        Index is datetime (month-end)
    
    Notes:
    ------
    Developed Factors cover Australia, Austria, Belgium, Canada, Switzerland,
    Germany, Denmark, Spain, Finland, France, Great Britain, Greece, 
    Hong Kong, Ireland, Italy, Japan, Netherlands, Norway, New Zealand,
    Portugal, Sweden, Singapore and United States.
    
    Source: Kenneth R. French Data Library
    https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
    """
    
    # Developed 5-Factor data URL
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Developed_5_Factors_CSV.zip"
    
    
    try:
        r = requests.get(url)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Failed to download Developed factors from Ken French library: {e}")
    
    # Open the ZIP file from bytes
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        # Find the CSV file
        csv_files = [name for name in z.namelist() if name.lower().endswith('.csv')]
        if not csv_files:
            raise ValueError("No CSV file found in the downloaded ZIP")
        
        csv_filename = csv_files[0]
        with z.open(csv_filename) as f:
            lines = f.read().decode('utf-8').splitlines()
    
    # Locate the header row
    header_idx = None
    for i, line in enumerate(lines):
        if "Mkt-RF" in line and "RF" in line:
            header_idx = i
            break
    
    if header_idx is None:
        raise ValueError("Could not find header row in Developed_5_Factors")
    
    header = lines[header_idx].strip().split(',')
    
    # Collect only the monthly data lines (lines starting with YYYYMM)
    data_lines = []
    for line in lines[header_idx + 1:]:
        stripped = line.strip()
        if not stripped:
            continue
        # Stop at annual data or other sections
        if not stripped[:6].isdigit():
            break
        data_lines.append(stripped)
    
    if not data_lines:
        raise ValueError("No monthly data found in Developed_5_Factors")
    
    # Create DataFrame
    df = pd.read_csv(io.StringIO("\n".join(data_lines)), names=header)
    
    # Parse date and set as index
    date_col = header[0]
    df[date_col] = pd.to_datetime(df[date_col], format="%Y%m")
    df.set_index(date_col, inplace=True)
    
    # Convert percentages to decimals
    df = df / 100
    # Uncomment the following lines to see summary info when testing
    # print(f"Loaded {len(df)} months of European factors")
    # print(f"Date range: {df.index[0].strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')}")
    
    return df

"""
# Uncomment this following block to test the function directly.

if __name__ == "__main__":
    # Test the function
    print("Testing Developed factor fetching...\n")
    df = fetch_ff5_monthly()
    print("\nFirst 5 rows:")
    print(df.head())
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nShape: {df.shape}")
    print(f"\nDate range: {df.index[0]} to {df.index[-1]}")
"""
