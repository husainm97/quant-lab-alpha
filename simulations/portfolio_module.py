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
        self.base_currency = 'USD'
        self.factor_region = 'Developed'

    def _price_to_monthly_returns(self, price_series: pd.Series) -> pd.Series:
        """
        Convert daily prices to monthly returns via compounding.
        """
        prices = price_series.dropna().copy()
        prices.index = pd.to_datetime(prices.index)
        daily_ret = prices.pct_change().dropna()
        monthly_ret = (1 + daily_ret).resample('ME').prod() - 1
        return monthly_ret.dropna()
    
    def _convert_to_base_currency(self, price_series: pd.Series, currency: str, target_currency: str) -> pd.Series:
        """
        Converts a price series from `currency` to `target_currency`, keeping all safeguards:
        - Flatten multi-index FX data
        - Reset index before merging
        - Merge on Date
        - Preserve datetime index
        """
        if currency == target_currency:
            return price_series.copy()

        fx_ticker = f"{currency}{target_currency}=X"
        fx_df = yf.download(fx_ticker,
                            start=price_series.index[0],
                            end=price_series.index[-1],
                            auto_adjust=True,
                            progress=False)

        # Flatten multi-index if present
        if isinstance(fx_df.columns, pd.MultiIndex):
            fx_df.columns = [col[0] for col in fx_df.columns]

        if 'Close' not in fx_df.columns:
            raise ValueError(f"FX data for {fx_ticker} has no Close column")

        fx_series = fx_df['Close'].copy()
        fx_series.index = pd.to_datetime(fx_series.index)

        # Reset index and merge on Date
        price_df = price_series.reset_index().rename(columns={'index': 'Date'})
        fx_df_reset = fx_series.reset_index().rename(columns={'Close': 'FX'})
        merged = pd.merge(price_df, fx_df_reset, on='Date', how='inner')

        # Conversion
        merged['converted'] = merged.iloc[:, 1] * merged['FX']

        # Restore datetime index
        converted = merged.set_index('Date')['converted']

        return converted

    def add_asset(self, ticker: str, weight: float, df: pd.DataFrame = None):
        if weight < 0:
            raise ValueError("Weight cannot be negative.")
        self.constituents[ticker] = weight

        if df is not None:
            # Extract price series
            if 'Adj Close' in df.columns:
                price = df['Adj Close']
            elif 'Close' in df.columns:
                price = df['Close']
            else:
                raise ValueError(f"No Close/Adj Close column in {ticker} data.")

            t = yf.Ticker(ticker)
            info = t.info
            currency = info.get('currency', 'USD')

            # USD copy for FF5
            prices_usd = price.copy() if currency == 'USD' else self._convert_to_base_currency(price, currency, target_currency='USD')

            # Portfolio base currency
            prices_base = price.copy() if currency == self.base_currency else self._convert_to_base_currency(price, currency, target_currency=self.base_currency)

            # Monthly returns based on base currency
            monthly_returns = self._price_to_monthly_returns(prices_base)

            # Store both
            self.data[ticker] = {
                'prices': prices_base,
                'prices_usd': prices_usd,
                'monthly_returns': monthly_returns
            }

    def remove_asset(self, ticker: str):
        """Remove an asset and its data from the portfolio."""
        if ticker in self.constituents:
            del self.constituents[ticker]
        if ticker in self.data:
            del self.data[ticker]

    def _normalize_weights(self):
        """Ensure total allocation sums to 1 (100%) unless leveraging."""
        raise NotImplementedError
        total = sum(self.constituents.values())
        if total > 0:
            self.constituents = {k: v / total for k, v in self.constituents.items()}

    def update_currency(self, new_currency):
        """Recompute prices by tracking the current state of the data."""
        old_currency = self.base_currency  # Where we are now
        self.base_currency = new_currency  # Where we are going
        
        for ticker, ent in self.data.items():
            # 1. Get the data in its CURRENT form (could be EUR, USD, etc.)
            current_prices = ent['prices']
            
            # 2. Convert from the OLD base to the NEW base
            # Note: We use old_currency here, NOT the ticker's native currency
            converted = self._convert_to_base_currency(current_prices, old_currency, new_currency)
            
            # 3. Update the storage
            ent['prices'] = converted
            ent['monthly_returns'] = self._price_to_monthly_returns(converted)


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
    
    def get_common_monthly_returns(self, use_usd: bool = False) -> pd.DataFrame:
        """
        Return a DataFrame of monthly returns for all constituents aligned to the
        maximal common intersection (max(start_dates), min(end_dates)).

        Parameters
        ----------
        ff5 : bool
            If True, use USD prices/returns for Fama-French 5-factor regression.
            Otherwise, use portfolio base currency.

        Returns
        -------
        pd.DataFrame
            Monthly returns for all assets.
        """

        if not self.constituents:
            raise ValueError("Portfolio has no constituents")

        series_list = []

        for t in self.constituents.keys():
            ent = self.data.get(t)
            if ent is None:
                raise ValueError(f"No data for {t}. Add data first.")

            # Pick series depending on ff5 flag
            if use_usd:
                if 'prices_usd' in ent:
                    price_series = ent['prices_usd']
                else:
                    raise ValueError(f"No USD prices stored for {t} required for FF5.")
                s = self._price_to_monthly_returns(price_series)
            else:
                if 'monthly_returns' in ent:
                    s = ent['monthly_returns']
                else:
                    # fallback to base prices
                    if 'prices' in ent:
                        s = self._price_to_monthly_returns(ent['prices'])
                    else:
                        raise ValueError(f"No base currency prices/returns for {t}.")

            s = s.copy()
            s.name = t
            series_list.append(s)

        # --- compute maximal common range ---
        start_common = max(s.index.min() for s in series_list)
        end_common = min(s.index.max() for s in series_list)

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
    