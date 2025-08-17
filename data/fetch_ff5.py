# data/fetch_ff5.py 

import pandas as pd
import zipfile
import io
import requests

def fetch_ff5_monthly():
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
    r = requests.get(url)
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, skiprows=3)
    return df

df = fetch_ff5_monthly()
print(df)
