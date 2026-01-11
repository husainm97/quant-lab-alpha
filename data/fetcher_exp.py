# data/fetcher.py
import pandas as pd
import yfinance as yf
import requests
import io
import zipfile
from registry import DATA_SOURCES
from datetime import datetime
from typing import Optional

"""
This is a planned expansion to fetcher.py with Stooq, FRED and Coingecko calls and normalisation to Yahoo format for compatibility.
Currently not in use. Suggestions for API-based data sources can be implemented with user's own keys.
"""

def _normalize_to_yahoo_format(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    Normalize any source dataframe to Yahoo Finance format:
    Columns: Open, High, Low, Close, Volume
    Index: DatetimeIndex
    """
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Column mapping for different sources
    column_mappings = {
        'stooq': {
            'Open': 'Open',
            'High': 'High', 
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume'
        },
        'fred': {
            # FRED typically has single value column
            'value': 'Close'
        },
        'coingecko': {
            'price': 'Close',
            'volume': 'Volume'
        }
    }
    
    # Apply source-specific normalization
    if source == 'stooq':
        # Stooq returns newest-first, so sort ascending
        df = df.sort_index()
        # Ensure proper column names (case-sensitive)
        df.columns = [col.capitalize() for col in df.columns]
        
    elif source == 'fred':
        # FRED has single value column with series ID as name, rename to Close
        if len(df.columns) == 1:
            # Get the actual value column (whatever it's named)
            value_col = df.columns[0]
            df['Close'] = df[value_col]
            df = df.drop(columns=[value_col])
        elif 'Close' not in df.columns:
            # If multiple columns but no Close, use first column
            value_col = df.columns[0]
            df['Close'] = df[value_col]
        
        # Add placeholder columns for OHLV format
        df['Open'] = df['Close']
        df['High'] = df['Close']
        df['Low'] = df['Close']
        df['Volume'] = 0
        
    elif source == 'coingecko':
        # CoinGecko already has price, add OHLV structure
        if 'price' in df.columns:
            df['Close'] = df['price']
            df['Open'] = df['price']
            df['High'] = df['price']
            df['Low'] = df['price']
            df = df.drop(columns=['price'])
        if 'volume' not in df.columns:
            df['Volume'] = 0
        else:
            df['Volume'] = df['volume']
            df = df.drop(columns=['volume'])
    
    # Ensure standard column order matching Yahoo
    standard_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    available_cols = [col for col in standard_cols if col in df.columns]
    
    # Add missing columns with NaN
    for col in standard_cols:
        if col not in df.columns:
            df[col] = float('nan')
    
    return df[standard_cols]


def fetch(key: str, start_date: Optional[str] = None, end_date: Optional[str] = None):
    """
    Fetch a dataset by key, normalized to Yahoo Finance format.
    
    Key examples:
        - 'FF/SMB' → Fama-French factors (CSV/ZIP)
        - 'Yahoo/AAPL' → Yahoo Finance ticker
        - 'Stooq/AAPL.US' → Stooq ticker
        - 'FRED/GDP' → FRED economic series
        - 'CoinGecko/bitcoin' → CoinGecko crypto
    
    Returns:
        pd.DataFrame with columns: Open, High, Low, Close, Volume
        Index: DatetimeIndex
    """
    # Set default date range
    if start_date is None:
        start_date = '1900-01-01'
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    
    # === YAHOO FINANCE ===
    if key.startswith("Yahoo/"):
        ticker = key.split("/", 1)[1]
        df = yf.download(ticker, start=start_date, end=end_date, 
                        auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError(f"No data returned from Yahoo for ticker {ticker}")
        return df
    
    # === STOOQ ===
    elif key.startswith("Stooq/"):
        ticker = key.split("/", 1)[1]
        # Stooq URL format: daily interval
        url = f"https://stooq.com/q/d/l/?s={ticker}&i=d"
        resp = requests.get(url)
        resp.raise_for_status()
        
        df = pd.read_csv(io.StringIO(resp.text))
        
        # Stooq uses 'Date' column
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
        
        df = _normalize_to_yahoo_format(df, 'stooq')
        
        # Filter by date range
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        return df
    
    # === FRED (Federal Reserve Economic Data) ===
    elif key.startswith("FRED/"):
        series_id = key.split("/", 1)[1]
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        resp = requests.get(url)
        resp.raise_for_status()
        
        df = pd.read_csv(io.StringIO(resp.text), parse_dates=['DATE'], index_col='DATE')
        
        # Remove any non-numeric values (FRED uses '.' for missing)
        df = df.replace('.', float('nan'))
        df = df.apply(pd.to_numeric, errors='coerce')
        
        # Drop rows with NaN values
        df = df.dropna()
        
        # Filter by date range BEFORE normalization to avoid empty df
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        df = df.loc[start_dt:end_dt]
        
        if df.empty:
            raise ValueError(f"No FRED data for {series_id} in date range {start_date} to {end_date}")
        
        df = _normalize_to_yahoo_format(df, 'fred')
        return df
    
    # === COINGECKO (Cryptocurrency) ===
    elif key.startswith("CoinGecko/"):
        coin_id = key.split("/", 1)[1]
        
        # Calculate days from start_date to end_date
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        days = (end_dt - start_dt).days
        
        # CoinGecko API (free tier now has strict limits)
        # Using demo API which has better rate limits
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': min(days, 365),  # Limit to 1 year for free tier
            'interval': 'daily'
        }
        
        # Add headers to avoid bot detection
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                raise ValueError(
                    f"CoinGecko API requires authentication now. "
                    f"Free tier is heavily rate-limited. "
                    f"Consider using alternative: Yahoo/{coin_id}-USD for major coins "
                    f"(e.g., Yahoo/BTC-USD, Yahoo/ETH-USD)"
                ) from e
            raise
        
        # Extract price and volume data
        if 'prices' not in data or not data['prices']:
            raise ValueError(f"No price data returned for {coin_id}")
            
        prices_df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        prices_df['timestamp'] = pd.to_datetime(prices_df['timestamp'], unit='ms')
        prices_df = prices_df.set_index('timestamp')
        
        if 'total_volumes' in data and data['total_volumes']:
            volumes_df = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
            volumes_df['timestamp'] = pd.to_datetime(volumes_df['timestamp'], unit='ms')
            volumes_df = volumes_df.set_index('timestamp')
            prices_df = prices_df.join(volumes_df)
        
        df = _normalize_to_yahoo_format(prices_df, 'coingecko')
        
        # Filter by exact date range
        df = df[(df.index >= start_dt) & (df.index <= end_dt)]
        return df
    
    # === REGISTRY-BASED SOURCES (Original) ===
    else:
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
                    start_idx = next(i for i, line in enumerate(lines) 
                                   if line.strip() and line.strip()[0].isdigit())
                    data = []
                    for line in lines[start_idx:]:
                        line = line.strip()
                        if not line:
                            break
                        data.append(line)
                    df = pd.read_csv(io.StringIO("\n".join(data)), 
                                   sep=r"\s+", index_col=0)
            return df
        
        elif dtype == "custom":
            raise NotImplementedError("Custom handler not implemented yet")
        else:
            raise ValueError(f"Unsupported data type: {dtype}")


# Example usage:
if __name__ == "__main__":
    # All return same format: Open, High, Low, Close, Volume with DatetimeIndex
    
    # Yahoo Finance
    aapl_yahoo = fetch("Yahoo/AAPL", start_date="2020-01-01", end_date="2023-12-31")
    print("Yahoo AAPL:")
    print(aapl_yahoo.head())
    print()
    
    # Stooq (alternative to Yahoo)
    aapl_stooq = fetch("Stooq/AAPL.US", start_date="2020-01-01", end_date="2023-12-31")
    print("Stooq AAPL:")
    print(aapl_stooq.head())
    print()
    
    # Not working
    # FRED economic data (try monthly series for better date coverage)
    #unrate = fetch("FRED/UNRATE", start_date="2020-01-01", end_date="2023-12-31")
    #print("FRED Unemployment Rate (monthly):")
    #print(unrate.head())
    #print()
    
    # For crypto, use Yahoo instead of CoinGecko (more reliable)
    btc = fetch("Yahoo/BTC-USD", start_date="2020-01-01", end_date="2023-12-31")
    print("Yahoo Bitcoin:")
    print(btc.head())