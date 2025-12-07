# data/fetch_ff5.py 

import pandas as pd
import zipfile
import io
import requests

def fetch_ff5_monthly():
    """
    Fetches Fama-French 5-Factor monthly data (2x3 CSV format),
    parses only the monthly rows, converts percentages to decimals,
    and returns a clean DataFrame with datetime index.
    """
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
    r = requests.get(url)
    
    # Open the ZIP file from bytes
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        # Find the CSV file
        csv_filename = next(name for name in z.namelist() if name.lower().endswith('.csv'))
        with z.open(csv_filename) as f:
            lines = f.read().decode('utf-8').splitlines()
    
    # Locate the header row
    header_idx = next(i for i, line in enumerate(lines) if "Mkt-RF" in line and "RF" in line)
    header = lines[header_idx].strip().split(',')
    
    # Collect only the monthly data lines (lines starting with YYYYMM)
    data_lines = []
    for line in lines[header_idx + 1:]:
        if not line.strip() or not line[:6].isdigit():
            break
        data_lines.append(line.strip())
    
    # Create DataFrame
    df = pd.read_csv(io.StringIO("\n".join(data_lines)), names=header)
    
    # Parse date and set as index
    df[header[0]] = pd.to_datetime(df[header[0]], format="%Y%m")
    df.set_index(header[0], inplace=True)
    
    # Convert percentages to decimals
    df = df / 100
    
    return df