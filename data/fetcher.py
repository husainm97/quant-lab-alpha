# data/fetcher.py

import pandas as pd
import yfinance as yf
import requests
import io
import zipfile

from data.registry import DATA_SOURCES

def fetch(key: str):
    """Fetch a dataset by key (like 'FF/SMB' or 'Yahoo/SP500')."""
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
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            with z.open(z.namelist()[0]) as f:
                df = pd.read_csv(f, skiprows=3)
        return df

    elif dtype == "yahoo":
        ticker = cfg["ticker"]
        return yf.download(ticker)

    elif dtype == "custom":
        # Placeholder: dynamically call a custom handler if you want
        raise NotImplementedError("Custom handler not implemented yet")

    else:
        raise ValueError(f"Unsupported data type: {dtype}")
