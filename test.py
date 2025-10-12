# test.py

'''
This is a temporary file to test the fetcher, references to local files may now be replaced using data.fetch(), any information on key-url-datatype may be noted in registry.py 
Nex, we test a full process of importing a UCITS S&P500 ETF, and calculating the factor loadings over it's whole history
This must be expanded to calculate "rolling" factor loadings with 10 year windows
'''

import data


def test_fetch():
    df_smb = data.fetch("FF/SMB")
    df_hml = data.fetch("FF/HML")
    
    print(df_smb)
    print(df_hml)
    
    print('now yahoo')
    sp500 = data.fetch("Yahoo/SP500")
    print(sp500)



def test_currency():
    """
    Runs a test case to validate the convert_currency_monthly function.
    """
    print("--- Running currency conversion test ---")

    # 1. Create a mock class for yfinance.download
    # This simulates downloading real-world data without a network call.
    class MockYFinance:
        def download(self, ticker, start, end, interval):
            # We'll return a pre-defined DataFrame based on the ticker.
            if ticker == "EURUSD=X":
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
        # 2. Create a sample asset returns series
        # The index has a different frequency and start date to test the alignment.
        asset_returns = pd.Series(
            [0.02, 0.015, -0.01, 0.03],
            index=[datetime(2023, 2, 28), datetime(2023, 3, 31), 
                   datetime(2023, 4, 30), datetime(2023, 5, 31)]
        )
        print("\nInput Asset Returns (EUR):")
        print(asset_returns)

        # 3. Manually calculate the expected FX returns and final converted returns
        # This confirms our understanding of the formula.
        fx_rates = pd.Series(
            [1.065, 1.080, 1.100, 1.095],
            index=[datetime(2023, 2, 1), datetime(2023, 3, 1), datetime(2023, 4, 1), datetime(2023, 5, 1)]
        )
        fx_returns = fx_rates.pct_change().dropna()

        # The alignment will start from Feb 2023
        expected_returns = (1 + asset_returns) * (1 + fx_returns.reindex(asset_returns.index)) - 1
        
        # 4. Call the function to be tested
        converted_returns = convert_currency_monthly(asset_returns, from_cur="EUR")

    print("\nExpected Converted Returns (USD):")
    print(expected_returns.round(4))
    
    print("\nActual Converted Returns from function (USD):")
    print(converted_returns.round(4))

    # 5. Assert that the outputs are identical
    pd.testing.assert_series_equal(converted_returns.round(4), expected_returns.round(4))
    
    print("\n--- Test Passed! ---")
    print("The function correctly converted the asset returns by compounding them with the FX returns.")

# Run the test
test_currency()
