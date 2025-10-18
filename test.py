# test.py

'''
This is a temporary file to test the fetcher, references to local files may now be replaced using data.fetch(), any information on key-url-datatype may be noted in registry.py 
'''

import data
from unittest.mock import patch, Mock
import yfinance as yf
import pandas as pd
from datetime import datetime
from src.utilities import *


def test_fetch():
    df_smb = data.fetch("FF/SMB")
    df_hml = data.fetch("FF/HML")
    
    print(df_smb)
    print(df_hml)
    
    print('now yahoo')
    sp500 = data.fetch("Yahoo/SP500")
    print(sp500)

# =================================================================
# 🎯 THE FUNCTION TO BE TESTED
# This version includes the CRUCIAL index alignment logic to match
# month-end asset returns with month-start FX rates.
# =================================================================
def convert_currency_monthly(asset_returns: pd.Series, from_cur: str) -> pd.Series:
    """
    Converts monthly asset returns denominated in 'from_cur' to USD.
    Assumes FX rates are fetched monthly.
    """
    fx_ticker = f"{from_cur}USD=X"
    
    # 1. Determine the date range needed for the FX rates.
    # We need the rate from one period before the first return up to the last return date.
    start_date = asset_returns.index.min() - pd.DateOffset(months=1)
    end_date = asset_returns.index.max() 
    
    # 2. Fetch the FX Rates (This uses the MockYFinance in the test context)
    # The interval="1mo" is used, but the mock provides specific dates.
    fx_data = yf.download(fx_ticker, start=start_date, end=end_date, interval="1mo")
    
    if fx_data.empty:
        return pd.Series([], dtype='float64') 

    fx_rates = fx_data['Adj Close'].sort_index()

    # 3. Calculate Monthly FX Returns
    # The return at date T is the return from T-1 to T.
    # With the mock data, this series will be indexed by the 1st of the month.
    fx_returns = fx_rates.pct_change().dropna()
    
    # 4. ALIGNMENT FIX: Match the length and index
    # Slice the FX returns to match the number of asset returns (removes any excess months)
    fx_returns = fx_returns.iloc[:len(asset_returns)] 
    
    # CRUCIAL: Force the FX returns index (1st of month) to match the asset returns 
    # index (end of month) for compounding. This assumes the FX return correctly
    # calculated the change for the period.
    fx_returns.index = asset_returns.index 

    # 5. Compound the Returns
    # Compounding formula: (1 + Asset Return) * (1 + FX Return) - 1
    converted_returns = (1 + asset_returns) * (1 + fx_returns) - 1
    
    return converted_returns

import pandas as pd
from datetime import datetime
from io import StringIO
import yfinance as yf
from unittest.mock import patch, Mock
# You must also define/import your 'convert_currency_monthly' function somewhere

def test_currency():
    """
    Runs a test case to validate the convert_currency_monthly function.
    """
    print("--- Running currency conversion test ---")

    # 1. Create a mock class for yfinance.download
    class MockYFinance:
        def download(self, ticker, start, end, interval):
            # We'll return a pre-defined DataFrame based on the ticker.
            if ticker == "EURUSD=X":
                # Rates are given for the 1st of the month
                data_string = """Date,Adj Close
2023-01-01,1.053
2023-02-01,1.065
2023-03-01,1.080
2023-04-01,1.100
2023-05-01,1.095
2023-06-01,1.088
"""
                df = pd.read_csv(StringIO(data_string), index_col='Date', parse_dates=True)
                return df
            else:
                return pd.DataFrame()

    # Use patch to replace the real yfinance.download with our mock
    with patch.object(yf, 'download', new=MockYFinance().download):
        # 2. Create a sample asset returns series (month-end dates)
        asset_returns_index = [
            datetime(2023, 2, 28), 
            datetime(2023, 3, 31), 
            datetime(2023, 4, 30), 
            datetime(2023, 5, 31)
        ]
        asset_returns = pd.Series(
            [0.02, 0.015, -0.01, 0.03],
            index=asset_returns_index
        )
        print("\nInput Asset Returns (EUR):")
        print(asset_returns)

        # 3. Manually calculate the expected FX returns and final converted returns
        
        # Get the FX rates (indexed by the 1st of the month)
        fx_rates = pd.Series(
            [1.053, 1.065, 1.080, 1.100, 1.095, 1.088], # Must include Jan 1st rate for Feb return calculation
            index=[datetime(2023, 1, 1), datetime(2023, 2, 1), datetime(2023, 3, 1), 
                   datetime(2023, 4, 1), datetime(2023, 5, 1), datetime(2023, 6, 1)]
        )
        
        # Calculate monthly FX returns (Feb 1st return corresponds to the return for January)
        fx_returns = fx_rates.pct_change().dropna()
        fx_returns = fx_returns.iloc[:4] # Take only the first 4 elements

        # 💥 FIX APPLIED HERE: Re-index the calculated FX returns to the month-end dates 
        # of the asset returns, assuming your convert_currency_monthly does the same.
        # This maps the FX return *for* the month to the *end date* of that month.
        fx_returns.index = asset_returns_index 

        # The alignment will start from Feb 2023
        expected_returns = (1 + asset_returns) * (1 + fx_returns) - 1
        
        # 4. Call the function to be tested
        # NOTE: This assumes 'convert_currency_monthly' handles the index alignment internally.
        converted_returns = convert_currency_monthly(asset_returns, from_cur="EUR")

    print("\nExpected Converted Returns (USD):")
    print(expected_returns.round(4))
    
    print("\nActual Converted Returns from function (USD):")
    print(converted_returns.round(4))

    # 5. Assert that the outputs are identical
    pd.testing.assert_series_equal(converted_returns.round(4), expected_returns.round(4))
    
    print("\n--- Test Passed! ---")
    print("The function correctly converted the asset returns by compounding them with the FX returns.")


# Test fetcher
test_fetch()

# Test currency converter
test_currency()
