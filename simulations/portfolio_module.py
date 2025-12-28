import sys
from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np
sys.path.append(str(Path(__file__).resolve().parent.parent))  # adds project/ to path
from data.fetcher import fetch
import yfinance as yf

class Portfolio:
    def __init__(self, name: str = "My Portfolio"):
        self.name = name
        self.constituents: Dict[str, float] = {}  # e.g., {"AAPL": 0.4, "GOOG": 0.6}
        self.leverage: float = 1.0
        self.interest_rate: float = 0.0
        self.cash: float = 0.0  # optional — for unallocated funds or interest accrual
        self.data: Dict[str, "pd.DataFrame"] = {}  # ticker -> fetched price/return DataFrame

    def _price_to_monthly_returns(self, price_series: pd.Series) -> pd.Series:
        """
        Convert daily prices to monthly returns via compounding.
        """
        prices = price_series.dropna().copy()
        prices.index = pd.to_datetime(prices.index)
        daily_ret = prices.pct_change().dropna()
        monthly_ret = (1 + daily_ret).resample('ME').prod() - 1
        return monthly_ret.dropna()

    def add_asset(self, ticker: str, weight: float, df: pd.DataFrame = None):
        """
        Add or update an asset weight in the portfolio.
        Optionally store its fetched data as a DataFrame.
        """
        if weight < 0:
            raise ValueError("Weight cannot be negative.")
        self.constituents[ticker] = weight
        if df is not None:
            # extract price series
            if 'Adj Close' in df.columns:
                price = df['Adj Close']
            elif 'Close' in df.columns:
                price = df['Close']
            else:
                raise ValueError(f"No Close/Adj Close column in {ticker} data.")
            
            t = yf.Ticker(ticker)
            info = t.info  # dictionary with metadata
            currency = info.get('currency', 'USD')
            if currency != 'USD':
                # Non-USD currency detected, here we pull the exchange data from yfinance to convert to USD for correct ff5 analysis
                
                forex_ticker = f'{currency}USD=X'
                # download EUR/USD FX data
                start_date = df.index[0]
                end_date = df.index[-1]

                eurusd_df = yf.download(forex_ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)

                # If it's a multi-index (common with auto_adjust), flatten
                if isinstance(eurusd_df.columns, pd.MultiIndex):
                    eurusd_df.columns = [col[0] for col in eurusd_df.columns]

                eurusd_df = eurusd_df[['Close']].copy()
                eurusd_df = eurusd_df.rename(columns={'Close': 'EURUSD'}).reset_index()

                # merge price DataFrame with FX rates
                price_df = price.reset_index().rename(columns={'index': 'Date'})
                merged = pd.merge(price_df, eurusd_df, on='Date', how='inner')

                # convert EUR → USD
                merged[ticker] = merged.iloc[:, 1] * merged['EURUSD']  # the ETF price column is the second column

                # restore datetime index
                merged['Date'] = pd.to_datetime(merged['Date'])
                price = merged.set_index('Date')[ticker]
                #print(f"Warning: Currency of {ticker} was converted from {currency} to USD.")

            monthly_returns = self._price_to_monthly_returns(price)
            self.data[ticker] = {
                'prices': df.copy(),
                'monthly_returns': monthly_returns
            }
        self._normalize_weights()


    def remove_asset(self, ticker: str):
        """Remove an asset and its data from the portfolio."""
        if ticker in self.constituents:
            del self.constituents[ticker]
        if ticker in self.data:
            del self.data[ticker]
        self._normalize_weights()

    def _normalize_weights(self):
        """Ensure total allocation sums to 1 (100%) unless leveraging."""
        total = sum(self.constituents.values())
        if total > 0:
            self.constituents = {k: v / total for k, v in self.constituents.items()}

    def total_allocation(self):
        """Return sum of weights (should be 1.0 for fully invested portfolios)."""
        return sum(self.constituents.values())

    def apply_leverage(self, leverage: float, interest_rate: float):
        """Set leverage and associated borrowing cost."""
        if leverage < 1:
            raise ValueError("Leverage must be >= 1.")
        self.leverage = leverage
        self.interest_rate = interest_rate

    def expected_return(self, asset_returns: Dict[str, float]) -> float:
        """Compute expected portfolio return given individual expected returns."""
        return sum(asset_returns[ticker] * weight for ticker, weight in self.constituents.items()) * self.leverage

    def variance(self, cov_matrix: np.ndarray, tickers: List[str]) -> float:
        """Compute portfolio variance using covariance matrix."""
        weights = np.array([self.constituents[t] for t in tickers])
        return self.leverage**2 * weights.T @ cov_matrix @ weights
    
    def get_common_monthly_returns(self) -> pd.DataFrame:
        """
        Return a DataFrame of monthly returns for all constituents aligned to the
        maximal common intersection (max(start_dates), min(end_dates)).
        Ensures each asset has monthly returns, computes them if only prices exist.
        """


        if not self.constituents:
            raise ValueError("Portfolio has no constituents")

        series_list = []

        for t in self.constituents.keys():
            ent = self.data.get(t)

            if ent is None:
                raise ValueError(f"No data for {t}. Add data first.")

            # handle dict with monthly_returns
            if isinstance(ent, dict) and 'monthly_returns' in ent:
                s = ent['monthly_returns']
            # handle raw price Series/DataFrame (legacy)
            elif isinstance(ent, pd.DataFrame):
                if 'Adj Close' in ent.columns:
                    price = ent['Adj Close']
                elif 'Close' in ent.columns:
                    price = ent['Close']
                else:
                    raise ValueError(f"No Close/Adj Close column for {t}")
                s = self._price_to_monthly_returns(price)
            elif isinstance(ent, pd.Series):
                s = self._price_to_monthly_returns(ent)
            else:
                raise ValueError(f"Cannot interpret data for {t}")

            s = s.copy()
            s.name = t
            series_list.append(s)

        # --- compute maximal common range ---
        start_common = max(s.index.min() for s in series_list)
        end_common = min(s.index.max() for s in series_list)

        #print(start_common, end_common)

        if start_common >= end_common:
            raise ValueError("No overlapping date range between assets.")

        # slice each series to common range and combine
        df = pd.concat([s.loc[start_common:end_common] for s in series_list], axis=1)

        # drop rows with any NaN
        df = df.dropna(how='any')

        if df.shape[0] < 12:
            raise ValueError(f"Common history too short: {df.shape[0]} months")

        # store the range for reference
        self._common_range = (start_common, end_common)

        return df



    def reset(self):
        """Clear the portfolio."""
        self.constituents.clear()
        self.data.clear()
        self.leverage = 1.0
        self.interest_rate = 0.0
        self.cash = 0.0


    def summary(self):
        """Readable portfolio summary."""
        return {
            "Name": self.name,
            "Constituents": self.constituents,
            "Leverage": self.leverage,
            "Interest Rate": self.interest_rate,
            "Total Allocation": self.total_allocation(),
        }
