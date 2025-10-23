# -----------------------------
# File: portfolio_bootstrap/bootstrap.py
# -----------------------------
"""Core bootstrap and simulation routines."""
import numpy as np
import pandas as pd
from typing import Optional, Callable, Sequence
from .utilities import apply_leverage
from .strategies import WithdrawalStrategy




def _circular_block_indices(length: int, block_len: int):
"""Yield a random start index for a circular block sample."""
start = np.random.randint(0, length)
idx = (np.arange(start, start + block_len) % length)
return idx




def block_bootstrap_series(returns: Sequence[float], n_periods: int, block_len: int) -> np.ndarray:
"""Generate a single bootstraped series of length `n_periods` using circular block bootstrap.


Args:
returns: original 1D sequence of returns (monthly decimals, e.g. 0.02 = 2%).
n_periods: length of the generated series (months).
block_len: block length in months.


Returns:
numpy array of length n_periods with sampled returns.
"""
arr = np.asarray(returns)
assert arr.ndim == 1, "returns must be a 1D sequence"
L = len(arr)
if block_len <= 0:
raise ValueError("block_len must be positive")


result = []
while len(result) < n_periods:
idx = _circular_block_indices(L, block_len)
result.extend(list(arr[idx]))
return np.array(result[:n_periods])




def _simulate_one(
returns_series: np.ndarray,
initial_capital: float,
leverage: float,
financing_rate_monthly: float,
withdraw_strategy: Optional[WithdrawalStrategy],
rebal_freq_months: int = 1
) -> pd.Series:
"""Run a single simulation applying leverage and withdrawal strategy.


Returns a pandas Series of portfolio values at each month (length = len(returns_series)).
"""
months = len(returns_series)
port = np.empty(months)
portfolio = initial_capital


for t in range(months):
# determine withdrawal at start of period (before returns) — strategy choice
if withdraw_strategy is not None:
withdrawal = withdraw_strategy.withdraw(portfolio=portfolio, month_index=t)
else:
withdrawal = 0.0


portfolio = max(portfolio - withdrawal, 0.0)


# apply leveraged return: portfolio_return = L*asset_ret - (L-1)*fin_rate
r = returns_series[t]
net_ret = leverage * r - max(0.0, leverage - 1.0) * financing_rate_monthly
portfolio = portfolio * (1.0 + net_ret)


port[t] = portfolio


# optional rebalancing could go here (placeholder). For now we assume implicit rebalancing maintains leverage.


return pd.Series(port)




def monte_carlo_simulation(
returns: Sequence[float],
n_periods: int,
n_sims: int = 1000,
block_len: int = 12,
initial_capital: float = 1_000_000.0,
leverage: float = 1.0,
financing_rate_annual: float = 0.0,
withdraw_strategy: Optional[WithdrawalStrategy] = None,
random_seed: Optional[int] = None,
progress: bool = False,
) -> pd.DataFrame:
return df
