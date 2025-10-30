"""
bootstrap.py

Generates 3 synthetic ETF return series, runs a block bootstrap simulation with a modular withdrawal strategy,
and plots portfolio wealth paths over a multi-year horizon with a translucent 1-sigma band around the mean.
Uses portfolio_module.Portfolio and withdrawal strategies from withdrawal_strategies.py.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Ensure local imports work
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from portfolio_module import Portfolio
from withdrawal_strategies import pct_of_initial_capital, pct_of_current_capital, fixed_amount

# ------------------------ PARAMETERS ------------------------
DAYS_PER_YEAR = 365
YEARS = 30
N_DAYS = DAYS_PER_YEAR * YEARS
N_ASSETS = 3
SEED = 42
N_BOOTSTRAP = 10000
BLOCK_SIZE = 365*5  # days per block
START_CAPITAL = 1_000_000.0
WEIGHTS = [0.2, 0.3, 0.5]
LEVERAGE = 1.25
ANNUAL_INTEREST = 0.05

# Select withdrawal strategy
strategy = pct_of_initial_capital(initial_capital=START_CAPITAL, rate=0.027, min_withdrawal=40_000, max_withdrawal=100_000)

# ETF specs: annual mu, sigma
ETF_SPECS = [
    (0.06, 0.12),
    (0.05, 0.05),
    (0.10, 0.20),
]
# ------------------------------------------------------------

rng = np.random.default_rng(SEED)

def make_ar1_returns(mu_annual, sigma_annual, n_days, phi=0.2):
    mu_daily = mu_annual / DAYS_PER_YEAR
    sigma_daily = sigma_annual / np.sqrt(DAYS_PER_YEAR)
    eps = rng.normal(0, sigma_daily, size=n_days)
    r = np.zeros(n_days)
    for t in range(1, n_days):
        r[t] = phi * r[t-1] + eps[t]
    r += (mu_daily - r.mean())
    return r

# Generate synthetic historical returns
returns_df = pd.DataFrame({f"ETF{i+1}": make_ar1_returns(mu, sig, N_DAYS)
                           for i, (mu, sig) in enumerate(ETF_SPECS)})

# Block bootstrap helper
def block_bootstrap_series(series: np.ndarray, block_size: int, n_samples: int, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    n = len(series)
    blocks = []
    k = int(np.ceil(n_samples / block_size))
    for _ in range(k):
        start = rng.integers(0, n - block_size + 1)
        blocks.append(series[start : start + block_size])
    return np.concatenate(blocks)[:n_samples]

# Initialize portfolio
weights = np.array(WEIGHTS) / np.sum(WEIGHTS)
portfolio_obj = Portfolio(name="Bootstrap Portfolio")
for i, w in enumerate(weights):
    portfolio_obj.add_asset(f"ETF{i+1}", float(w))
portfolio_obj.apply_leverage(LEVERAGE, ANNUAL_INTEREST)

daily_interest_cost = ANNUAL_INTEREST * (LEVERAGE - 1) / DAYS_PER_YEAR

# Run block bootstrap simulations
wealth_paths = np.zeros((N_BOOTSTRAP, N_DAYS + 1))
wealth_paths[:, 0] = START_CAPITAL

for b in range(N_BOOTSTRAP):
    resampled = np.column_stack([block_bootstrap_series(returns_df.iloc[:, j].values, BLOCK_SIZE, N_DAYS, rng)
                                 for j in range(N_ASSETS)])
    daily_port_returns = (resampled @ weights) * LEVERAGE - daily_interest_cost

    w = START_CAPITAL
    for t in range(N_DAYS):
        w = w * (1 + daily_port_returns[t])
        # apply withdrawal strategy annually
        if t % DAYS_PER_YEAR == 0 and t > 0:
            w -= strategy(w)
        wealth_paths[b, t+1] = w

# Compute statistics
mean_path = wealth_paths.mean(axis=0)
std_path = wealth_paths.std(axis=0)
final_wealth = wealth_paths[:, -1]
p16, p50, p84 = np.percentile(final_wealth, [16, 50, 84])

# Plot
plt.figure(figsize=(10,6))
xs = np.arange(N_DAYS+1)
for i in range(min(50, N_BOOTSTRAP)):
    plt.plot(xs, wealth_paths[i], alpha=0.08, linewidth=0.8)
plt.plot(xs, mean_path, label='Mean wealth', linewidth=2)
plt.fill_between(xs, mean_path - std_path, mean_path + std_path, alpha=0.25, label='±1 σ')

plt.title('Block Bootstrap: Portfolio Wealth Paths')
plt.xlabel('Days')
plt.ylabel('Wealth')
plt.grid(alpha=0.3)
#plt.legend()

txt = (
    f"Start: ${START_CAPITAL:,.0f}\n"
    f"Mean final: ${mean_path[-1]:,.0f}\n"
    f"P16: ${p16:,.0f}  P50: ${p50:,.0f}  P84: ${p84:,.0f}\n"
    f"Leverage: {LEVERAGE}  Interest (annual): {ANNUAL_INTEREST:.2%}  Block size: {BLOCK_SIZE}d"
)
plt.annotate(txt, xy=(0.02,0.98), xycoords='axes fraction', va='top', fontsize=9, bbox=dict(boxstyle='round', alpha=0.12))

plt.tight_layout()
plt.show()

# Summary
print("--- Summary ---")
print(f"Start capital: ${START_CAPITAL:,.0f}")
print(f"Mean final wealth: ${mean_path[-1]:,.2f}")
print(f"Median final wealth: ${p50:,.2f}")
print(f"P16 / P84: ${p16:,.2f} / ${p84:,.2f}")
print(f"Number of bootstrap trials: {N_BOOTSTRAP}")

res_df = pd.DataFrame({'final_wealth': final_wealth})
#res_df.to_csv('bootstrap_final_wealth.csv', index=False)
#print('Saved final-wealth samples to bootstrap_final_wealth.csv')
