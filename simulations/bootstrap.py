"""
bootstrap.py

Generates 3 synthetic ETF return series, runs a block bootstrap simulation with a modular withdrawal strategy,
and plots portfolio wealth paths over a multi-year horizon with a translucent 1-sigma band around the mean.
Uses portfolio_module.Portfolio and withdrawal strategies from withdrawal_strategies.py.
"""

# bootstrap.py

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
    fixed_pct_initial,
    pct_of_current_capital,
    guardrail_pct,
    bucket_strategy
)

def run_simulation(
    portfolio_obj: Portfolio = None,
    returns_df: pd.DataFrame = None,
    config: dict = None
):
    """
    Run a block-bootstrap Monte Carlo simulation using monthly returns.

    Args:
        portfolio_obj: Portfolio object (tickers, weights, leverage, optional price data)
        returns_df: Optional precomputed monthly returns DataFrame
        config: Dict with simulation params

    Returns:
        all_results: dict of results per strategy
        fig: matplotlib figure
    """

    if config is None:
        config = {}

    # ------------------------
    # Simulation parameters
    # ------------------------
    MONTHS_PER_YEAR = 12
    SEED = config.get("SEED", 42)
    N_BOOTSTRAP = config.get("N_BOOTSTRAP", 2000)
    BLOCK_SIZE = config.get("BLOCK_SIZE", 60)  # 5 years in months
    START_CAPITAL = config.get("START_CAPITAL", 1_000_000)
    TARGET_WEALTH = config.get("TARGET_WEALTH", 2_000_000)

    rng = np.random.default_rng(SEED)

    print("Running sim!")

    # ------------------------
    # Prepare returns
    # ------------------------
    if returns_df is not None:
        returns_df = returns_df.copy()
        tickers = list(returns_df.columns)
        N_ASSETS = len(tickers)
        N_MONTHS = returns_df.shape[0]

    elif portfolio_obj is not None:
        # get monthly returns aligned to maximal common range

        returns_df = portfolio_obj.get_common_monthly_returns()
        tickers = list(returns_df.columns)
        N_ASSETS = len(tickers)
        N_MONTHS = returns_df.shape[0]

    else:
        raise ValueError("Provide either portfolio_obj or returns_df")

    print("Returns ready!")
    # ------------------------
    # Weights and leverage
    # ------------------------
    print(portfolio_obj)
    weights = np.array([portfolio_obj.constituents[t] for t in tickers]) if portfolio_obj else np.ones(N_ASSETS)/N_ASSETS
    leverage = portfolio_obj.leverage if portfolio_obj else 1.0
    interest_rate = portfolio_obj.interest_rate if portfolio_obj else 0.0
    monthly_interest = interest_rate * (leverage - 1) / MONTHS_PER_YEAR
    print(f'Using leverage: {leverage}')
    # ------------------------
    # Block bootstrap helper
    # ------------------------
    def block_bootstrap(series, block_size, n_samples):
        n = len(series)
        block_size = min(block_size, max(1, n//2))
        blocks = []
        k = int(np.ceil(n_samples / block_size))
        for _ in range(k):
            start = rng.integers(0, n - block_size + 1)
            blocks.append(series[start:start+block_size])
        return np.concatenate(blocks)[:n_samples]

    # ------------------------
    # Generate portfolio return paths
    # ------------------------
    all_paths = np.zeros((N_BOOTSTRAP, N_MONTHS, N_ASSETS))
    for b in range(N_BOOTSTRAP):
        all_paths[b] = np.column_stack([
            block_bootstrap(returns_df[col].values, BLOCK_SIZE, N_MONTHS)
            for col in tickers
        ])
    port_returns = (all_paths @ weights) * leverage - monthly_interest

    # ------------------------
    # Withdrawal strategies
    # ------------------------
    strategies = {
        "Fixed 4%": fixed_pct_initial(START_CAPITAL, 0.04),
        "Variable 4%": pct_of_current_capital(0.04),
        "Guardrail 2.5-5%": guardrail_pct(0.025, 0.05),
        "Bucket Strategy": bucket_strategy(cash_years=3, annual_withdrawal=50_000)
    }

    all_results = {}
    xs = np.arange(N_MONTHS + 1)

    # ------------------------
    # Run simulations
    # ------------------------
    for name, strategy in strategies.items():
        wealth_paths = np.zeros((N_BOOTSTRAP, N_MONTHS + 1))
        wealth_paths[:, 0] = START_CAPITAL
        for b in range(N_BOOTSTRAP):
            w = START_CAPITAL
            for t in range(N_MONTHS):
                w *= 1 + port_returns[b, t]
                if t % MONTHS_PER_YEAR == 0 and t > 0:
                    w -= strategy(w)
                wealth_paths[b, t+1] = w

        final_wealth = wealth_paths[:, -1]
        terminated = (wealth_paths < 0).any(axis=1)

        all_results[name] = {
            "wealth_paths": wealth_paths,
            "mean_path": wealth_paths.mean(axis=0),
            "std_path": wealth_paths.std(axis=0),
            "final_wealth": final_wealth,
            "terminated": terminated,
            "fail_rate": terminated.mean()*100,
            "p16": np.percentile(final_wealth, 16),
            "p50": np.percentile(final_wealth, 50),
            "p84": np.percentile(final_wealth, 84),
            "prob_target": (final_wealth >= TARGET_WEALTH).mean()*100
        }

    # ------------------------
    # Plot
    # ------------------------
    n_strats = len(strategies)
    n_cols = 2
    n_rows = int(np.ceil(n_strats / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16,12), sharex=True)
    axes = axes.flatten()

    for ax, (name, res) in zip(axes, all_results.items()):
        wp = res["wealth_paths"]
        ax.fill_between(xs, 0, -START_CAPITAL*0.5, color='red', alpha=0.1)
        for i in range(min(50, N_BOOTSTRAP)):
            ax.plot(xs, np.where(wp[i]>=0, wp[i], np.nan), color='blue', alpha=0.08)
            ax.plot(xs, np.where(wp[i]<0, wp[i], np.nan), '--', color='red', alpha=0.15)
        ax.plot(xs, res["mean_path"], color='black', linewidth=2)
        ax.fill_between(xs,
                        res["mean_path"]-res["std_path"],
                        res["mean_path"]+res["std_path"],
                        alpha=0.2)
        txt = (
            f"Fail rate: {res['fail_rate']:.1f}%\n"
            f"P16/P50/P84: ${res['p16']:,.0f}/${res['p50']:,.0f}/${res['p84']:,.0f}\n"
            f"Prob ≥ ${TARGET_WEALTH:,}: {res['prob_target']:.1f}%"
        )
        ax.annotate(txt, xy=(0.02,0.95), xycoords='axes fraction', va='top', fontsize=9,
                    bbox=dict(boxstyle='round', alpha=0.1))
        ax.set_title(name)
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
        ax.grid(alpha=0.3)

    for ax in axes[n_strats:]:
        ax.axis('off')
    axes[-1].set_xlabel("Months")
    plt.tight_layout()
    return all_results, fig
