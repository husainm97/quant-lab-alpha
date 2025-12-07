import sys
from pathlib import Path
from typing import List, Dict
import numpy as np
sys.path.append(str(Path(__file__).resolve().parent.parent))  # adds project/ to path
from data.fetcher import fetch

class Portfolio:
    def __init__(self, name: str = "My Portfolio"):
        self.name = name
        self.constituents: Dict[str, float] = {}  # e.g., {"AAPL": 0.4, "GOOG": 0.6}
        self.leverage: float = 1.0
        self.interest_rate: float = 0.0
        self.cash: float = 0.0  # optional — for unallocated funds or interest accrual
        self.data: Dict[str, "pd.DataFrame"] = {}  # ticker -> fetched price/return DataFrame

    def add_asset(self, ticker: str, weight: float, df: "pd.DataFrame" = None):
        """
        Add or update an asset weight in the portfolio.
        Optionally store its fetched data as a DataFrame.
        """
        if weight < 0:
            raise ValueError("Weight cannot be negative.")
        self.constituents[ticker] = weight
        if df is not None:
            self.data[ticker] = df
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
