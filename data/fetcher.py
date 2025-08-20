# data/fetcher.py

import pandas as pd
import yfinance as yf
import requests
import io
import zipfile

from data.registry import DATA_SOURCES

def fetch(key: str):
    """
    Fetch a dataset by key (like 'FF/SMB' or 'Yahoo/SP500')
    Key syntax is defined in ./registry.py
    Currently supports csv, zip, yahoo
    Should add txt support in next expansion
    """
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
                raw = f.read().decode("latin-1")   # decode bytes -> string
                lines = raw.splitlines()
                start_idx = next(i for i, line in enumerate(lines) if line.strip() and line.strip()[0].isdigit())
                data = []
                # skip first 2 lines (or however many headers the file has)
                for line in lines[start_idx:]:
                    line = line.strip()
                    print(line)
                    if not line:   # stop at first blank line
                        break
                    '''
                    first_col = line.split()[0]
                    if not first_col.replace('.', '', 1).isdigit():  # stop at footer/non-numeric
                        break
                    '''
                    data.append(line)

                # load into DataFrame
                df = pd.read_csv(
                    io.StringIO("\n".join(data)),
                    sep=r"\s+",      # split on whitespace
                    index_col=0
                )
        return df

    elif dtype == "yahoo":
        ticker = cfg["ticker"]
        return yf.download(ticker)

    elif dtype == "custom":
        # Placeholder: dynamically call a custom handler if you want
        raise NotImplementedError("Custom handler not implemented yet")

    else:
        raise ValueError(f"Unsupported data type: {dtype}")
