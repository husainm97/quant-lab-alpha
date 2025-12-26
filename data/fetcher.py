# data/fetcher.py

import pandas as pd
import yfinance as yf
import requests
import io
import zipfile
from data.registry import DATA_SOURCES
from datetime import datetime

def fetch(key: str):
    """
    Fetch a dataset by key.
    Key examples:
        - 'FF/SMB' → CSV/ZIP sources
        - 'Yahoo/AAPL' → dynamic Yahoo ticker download
    """
    # Check if this is a dynamic Yahoo ticker request
    if key.startswith("Yahoo/"):
        ticker = key.split("/")[1]

        # Ask user or config for date range
        start_date = '1900-01-01'  # or input from config
        end_date = datetime.today().strftime('%Y-%m-%d')

        # Download daily data from Yahoo
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError(f"No data returned from Yahoo for ticker {ticker}")
        return df
        # Otherwise, look up in DATA_SOURCES
        if key not in DATA_SOURCES:
            raise KeyError(f"Unknown data key: {key}")

    cfg = DATA_SOURCES[key]
    dtype = cfg["type"]

    if dtype == "csv":
        resp = requests.get(cfg["url"])
        resp.raise_for_status()
        return pd.read_csv(io.StringIO(resp.text))
    
    elif dtype == "zip":
        r = requests.get(cfg["url"])
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            with z.open(z.namelist()[0]) as f:
                raw = f.read().decode("latin-1")
                lines = raw.splitlines()
                start_idx = next(i for i, line in enumerate(lines) if line.strip() and line.strip()[0].isdigit())
                data = []
                for line in lines[start_idx:]:
                    line = line.strip()
                    if not line:
                        break
                    data.append(line)
                df = pd.read_csv(io.StringIO("\n".join(data)), sep=r"\s+", index_col=0)
        return df

    elif dtype == "custom":
        raise NotImplementedError("Custom handler not implemented yet")

    else:
        raise ValueError(f"Unsupported data type: {dtype}")

