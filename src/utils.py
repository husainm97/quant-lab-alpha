import pandas as pd
import numpy as np
import requests
import zipfile
import io
import yfinance as yf

# Placeholder functions, build in progress

"""
TO DO: 
Add a currenry converter to remove currency fluctuations
""" 
def fetch_ff5_monthly():
    """
    Downloads Fama-French 5-factor monthly data from Ken French's site.
    Returns a cleaned DataFrame with date as index and decimal returns.
    """
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    fname = z.namelist()[0]
    
    with z.open(fname) as f:
        df = pd.read_csv(f, skiprows=3)

    # Clean up and parse
    df = df.rename(columns={df.columns[0]: "Date"})
    df = df[df["Date"].str.match(r"\d{6}")]
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m")
    df = df.set_index("Date")
    
    # Convert to decimal returns
    df = df.astype(float) / 100.0
    return df

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

def regress_ff5(portfolio_returns, ff_factors):
    """
    Runs OLS regression of portfolio excess returns on FF5 factors.
    Returns the regression coefficients and R².
    """
    import statsmodels.api as sm
    excess_returns = portfolio_returns - ff_factors['RF']
    X = ff_factors.drop(columns="RF")
    X = sm.add_constant(X)
    model = sm.OLS(excess_returns, X).fit()
    return model

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

    # Get FX data (monthly close)
    fx = yf.download(ticker,
                     start=data.index.min(),
                     end=data.index.max(),
                     interval="1mo")["Adj Close"]

    # Align FX index to monthly periods
    fx.index = fx.index.to_period("M").to_timestamp()

    # Compute FX monthly returns
    fx_ret = fx.pct_change().dropna()

    # Align and multiply returns
    aligned_asset, aligned_fx = data.align(fx_ret, join="inner")
    return (1 + aligned_asset) * (1 + aligned_fx) - 1
