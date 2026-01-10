import pandas as pd
import numpy as np
import requests
import zipfile
import io
import yfinance as yf
from typing import Sequence
from datetime import datetime
from unittest.mock import patch
from io import StringIO

"""
This is a general library to move commonly used functions into. 
""" 

def download_yahoo_prices(tickers, start, end):
    """
    Downloads adjusted close prices for a list of tickers.
    """
    data = yf.download(tickers, start=start, end=end)["Adj Close"]
    return data

def compute_monthly_returns(price_df):
    """
    Computes monthly log returns from a price DataFrame.
    """
    monthly_prices = price_df.resample("M").last()
    returns = np.log(monthly_prices / monthly_prices.shift(1)).dropna()
    return returns

def align_returns(portfolio_returns, ff_factors):
    """
    Aligns the portfolio returns with FF factor data by date.
    """
    df = pd.concat([portfolio_returns, ff_factors], axis=1, join="inner").dropna()
    return df

def convert_currency_monthly(data: pd.Series, from_cur: str, to_cur: str = "USD") -> pd.Series:
    """
    Convert a time series of monthly returns from one currency into another.

    Parameters
    ----------
    data : pd.Series
        Monthly returns in 'from_cur' currency, indexed by datetime.
    from_cur : str
        Source currency code (e.g., 'EUR', 'GBP', 'JPY').
    to_cur : str, optional
        Target currency code (default 'USD').

    Returns
    -------
    pd.Series
        Returns converted into 'to_cur'.
    """

    if from_cur == to_cur:
        return data

    # Yahoo ticker for FX pair, e.g. "EURUSD=X"
    ticker = f"{from_cur}{to_cur}=X"

    # CRITICAL FIX 1: Fetch rates starting ONE MONTH BEFORE the first asset return 
    # to ensure we capture the initial starting FX rate for the first period's return.
    start_date = data.index.min() - pd.DateOffset(months=1)
    end_date = data.index.max()

    # Get FX data (monthly close)
    fx_data = yf.download(ticker,
                          start=start_date,
                          end=end_date,
                          interval="1mo")
    
    if fx_data.empty:
        return pd.Series([], dtype='float64') 

    fx_rates = fx_data["Adj Close"].sort_index()

    # Compute FX monthly returns (indexed by the 1st of the month, starting with the 2nd month)
    fx_ret = fx_rates.pct_change().dropna()

    # CRITICAL FIX 2: Align the length and force the index to match the asset returns index.
    # This assumes the chronological order is correct, which it is in this monthly context.
    
    # Slice the FX returns to match the number of asset returns (removes any excess months)
    fx_ret = fx_ret.iloc[:len(data)]
    
    # Force the FX returns index (e.g., 2023-02-01) to match the asset returns index (e.g., 2023-02-28)
    fx_ret.index = data.index

    # Align (which is now just a check since indices are forced) and multiply returns
    converted_returns = (1 + data) * (1 + fx_ret) - 1
    
    return converted_returns
    

def apply_leverage(returns: Sequence[float], leverage: float, financing_rate_annual: float = 0.0):
    """
    Apply leverage to a sequence of periodic returns and return net leveraged returns.
    Formula used: net_ret = L*r - (L-1)*fin_rate_periodic

    Args:
    returns: sequence of periodic returns (decimals).
    leverage: leverage factor L.
    financing_rate_annual: annual financing/carry rate.

    Returns:
    numpy array of net periodic returns after leverage cost.
    """
    arr = np.asarray(returns, dtype=float)
    fr = (1.0 + financing_rate_annual) ** (1.0 / 12.0) - 1.0
    net = leverage * arr - max(0.0, leverage - 1.0) * fr
    return net



