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
