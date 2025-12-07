"""
bootstrap_comparison.py

Block bootstrap Monte Carlo simulation of a portfolio under multiple withdrawal strategies.
Generates synthetic ETF returns, applies leverage, and visualizes wealth paths with
mean, sigma bands, and risk of ruin.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import matplotlib.ticker as mtick

# Ensure local imports work
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from portfolio_module import Portfolio
from withdrawal_strategies import (
    fixed_pct_initial,    # 4% of initial
    pct_of_current_capital, # variable 4%
    guardrail_pct,          # 2.5–5% guardrails
    bucket_strategy # cash bucket
)

# ------------------------ PARAMETERS ------------------------
DAYS_PER_YEAR = 365
YEARS = 30
N_DAYS = DAYS_PER_YEAR * YEARS
N_ASSETS = 3
SEED = 42
N_BOOTSTRAP = 5000  # keep lower for faster plotting
BLOCK_SIZE = 365*5
START_CAPITAL = 1_000_000.0
WEIGHTS = [0.2, 0.3, 0.5]
LEVERAGE = 1.25
ANNUAL_INTEREST = 0.05

# Target for probability metrics
TARGET_WEALTH = 2_000_000

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
returns_df = pd.DataFrame({
    f"ETF{i+1}": make_ar1_returns(mu, sig, N_DAYS)
    for i, (mu, sig) in enumerate(ETF_SPECS)
})

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

# Initialize portfolio object
weights = np.array(WEIGHTS) / np.sum(WEIGHTS)
portfolio_obj = Portfolio(name="Bootstrap Portfolio")
for i, w in enumerate(weights):
    portfolio_obj.add_asset(f"ETF{i+1}", float(w))
portfolio_obj.apply_leverage(LEVERAGE, ANNUAL_INTEREST)

daily_interest_cost = ANNUAL_INTEREST * (LEVERAGE - 1) / DAYS_PER_YEAR

# ------------------------ STRATEGIES ------------------------
strategies = {
    "Fixed 4%": fixed_pct_initial(initial_capital=START_CAPITAL, rate=0.04),
    "Variable 4%": pct_of_current_capital(rate=0.04),
    "Guardrail 2.5-5%": guardrail_pct(min_rate=0.025, max_rate=0.05),
    "Bucket Strategy": bucket_strategy(cash_years=3, annual_withdrawal=50_000)
}

# ------------------------ RUN SIMULATIONS ------------------------
all_results = {}

# ------------------------ GENERATE RETURN PATHS ONCE ------------------------
all_paths = np.zeros((N_BOOTSTRAP, N_DAYS, N_ASSETS))
for b in range(N_BOOTSTRAP):
    all_paths[b] = np.column_stack([
        block_bootstrap_series(returns_df.iloc[:, j].values, BLOCK_SIZE, N_DAYS, rng)
        for j in range(N_ASSETS)
    ])
daily_port_returns_all = (all_paths @ weights) * LEVERAGE - daily_interest_cost  # shape: (N_BOOTSTRAP, N_DAYS)

# ------------------------ APPLY STRATEGIES ------------------------
all_results = {}

for strat_name, strategy in strategies.items():
    print(f"\nSimulating strategy: {strat_name}")
    wealth_paths = np.zeros((N_BOOTSTRAP, N_DAYS + 1))
    wealth_paths[:, 0] = START_CAPITAL

    for b in range(N_BOOTSTRAP):
        w = START_CAPITAL
        for t in range(N_DAYS):
            w = w * (1 + daily_port_returns_all[b, t])
            if t % DAYS_PER_YEAR == 0 and t > 0:
                w -= strategy(w)
            wealth_paths[b, t+1] = w


    # Compute stats
    mean_path = wealth_paths.mean(axis=0)
    std_path = wealth_paths.std(axis=0)
    final_wealth = wealth_paths[:, -1]
    terminated = (wealth_paths < 0).any(axis=1)
    fail_rate = terminated.mean() * 100
    p16, p50, p84 = np.percentile(final_wealth, [16, 50, 84])
    prob_target = (final_wealth >= TARGET_WEALTH).mean() * 100

    all_results[strat_name] = {
        "wealth_paths": wealth_paths,
        "mean_path": mean_path,
        "std_path": std_path,
        "final_wealth": final_wealth,
        "terminated": terminated,
        "fail_rate": fail_rate,
        "p16": p16,
        "p50": p50,
        "p84": p84,
        "prob_target": prob_target
    }

# ------------------------ PLOTTING ------------------------
n_strategies = len(strategies)
n_cols = 2
n_rows = 2  # 2x2 grid

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 12), sharex=True, sharey=False)
axes = axes.flatten()
xs = np.arange(N_DAYS + 1)

for ax, (strat_name, res) in zip(axes, all_results.items()):
    wp = res["wealth_paths"]
    terminated = res["terminated"]

    # Danger zone
    ax.fill_between(xs, 0, -START_CAPITAL*0.5, color='red', alpha=0.1)

    # Sample paths
    for i in range(min(50, N_BOOTSTRAP)):
        ax.plot(xs, np.where(wp[i]>=0, wp[i], np.nan), color='blue', alpha=0.08)
        ax.plot(xs, np.where(wp[i]<0, wp[i], np.nan), '--', color='red', alpha=0.15)

    # Mean ±1σ
    ax.plot(xs, res["mean_path"], label='Mean wealth', color='black', linewidth=2)
    ax.fill_between(xs,
                    res["mean_path"] - res["std_path"],
                    res["mean_path"] + res["std_path"],
                    alpha=0.2, label='±1σ')

    # Annotate key metrics directly
    txt = (
        f"Fail rate: {res['fail_rate']:.1f}%\n"
        f"P16/P50/P84: ${res['p16']:,.0f}/${res['p50']:,.0f}/${res['p84']:,.0f}\n"
        f"Prob ≥ ${TARGET_WEALTH:,}: {res['prob_target']:.1f}%"
    )
    ax.annotate(txt, xy=(0.02, 0.95), xycoords='axes fraction', va='top', fontsize=9,
                bbox=dict(boxstyle='round', alpha=0.1))

    ax.set_title(strat_name, fontsize=12)
    ax.set_ylabel('Wealth ($)')
    ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))

# Hide any unused subplots
for ax in axes[n_strategies:]:
    ax.axis('off')

axes[-1].set_xlabel('Days')
plt.tight_layout()
plt.show()

# ------------------------ SUMMARY ------------------------
for strat_name, res in all_results.items():
    print(f"\n--- {strat_name} ---")
    print(f"Failure rate: {res['fail_rate']:.2f}%")
    print(f"Median final wealth: ${res['p50']:.0f}")
    print(f"P16 / P84: ${res['p16']:.0f} / ${res['p84']:.0f}")
    print(f"Probability final wealth >= ${TARGET_WEALTH:,}: {res['prob_target']:.1f}%")
