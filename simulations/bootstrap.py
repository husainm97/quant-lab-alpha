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

def run_simulation(portfolio_obj: Portfolio = None, returns_df: pd.DataFrame = None, config=None):
    """
    Run a block bootstrap Monte Carlo simulation for a portfolio.

    Args:
        portfolio_obj: Portfolio object with tickers, weights, leverage, and optional price data.
        returns_df: Optional DataFrame with precomputed returns (columns = tickers, index = dates).
        config: Optional simulation config dict.

    Returns:
        all_results: dict of simulation results per withdrawal strategy
        fig: matplotlib figure with wealth paths
    """
    if config is None:
        config = {}

    DAYS_PER_YEAR = config.get("DAYS_PER_YEAR", 252)
    YEARS = config.get("YEARS", 30)
    N_DAYS = DAYS_PER_YEAR * YEARS
    SEED = config.get("SEED", 42)
    N_BOOTSTRAP = config.get("N_BOOTSTRAP", 2000)
    BLOCK_SIZE = config.get("BLOCK_SIZE", 252 * 5)
    START_CAPITAL = config.get("START_CAPITAL", 1_000_000.0)
    TARGET_WEALTH = config.get("TARGET_WEALTH", 2_000_000)

    rng = np.random.default_rng(SEED)

    if returns_df is not None:
        # Use provided returns directly
        tickers = list(returns_df.columns)
        N_ASSETS = len(tickers)
        returns_df = returns_df.copy()
    elif portfolio_obj is not None:
        tickers = list(portfolio_obj.constituents.keys())
        N_ASSETS = len(tickers)
        returns_df = pd.DataFrame()

        for ticker in tickers:
            df = portfolio_obj.data.get(ticker)
            if df is None:
                df = fetch(f"Yahoo/{ticker}")
                portfolio_obj.data[ticker] = df

            if 'Adj Close' in df.columns:
                price_series = df['Adj Close']
            elif 'Close' in df.columns:
                price_series = df['Close']
            else:
                raise ValueError(f"Ticker {ticker} has no Close or Adj Close column")

            returns_df[ticker] = price_series.pct_change().dropna()/100

        if returns_df.empty:
            raise ValueError("No valid return data found for portfolio tickers")
    else:
        raise ValueError("Either a portfolio object or a returns DataFrame must be provided")

    weights = np.array([portfolio_obj.constituents[t] for t in tickers]) if portfolio_obj else np.ones(N_ASSETS) / N_ASSETS
    leverage = portfolio_obj.leverage if portfolio_obj else 1.0
    interest_rate = portfolio_obj.interest_rate if portfolio_obj else 0.0

    # -------------------
    # Block bootstrap helper
    # -------------------
    def block_bootstrap_series(series, block_size, n_samples, rng_local=None):
        if rng_local is None:
            rng_local = np.random.default_rng()
        n = len(series)
        if n == 0:
            raise ValueError("Empty series provided to block bootstrap")
        block_size = min(block_size, n)
        blocks = []
        k = int(np.ceil(n_samples / block_size))
        for _ in range(k):
            start = rng_local.integers(0, n - block_size + 1)
            blocks.append(series[start:start + block_size])
        return np.concatenate(blocks)[:n_samples]

    # -------------------
    # Generate bootstrap paths
    # -------------------
    all_paths = np.zeros((N_BOOTSTRAP, N_DAYS, N_ASSETS))
    for b in range(N_BOOTSTRAP):
        all_paths[b] = np.column_stack([
            block_bootstrap_series(returns_df[col].values, BLOCK_SIZE, N_DAYS, rng)
            for col in tickers
        ])

    daily_port_returns_all = (all_paths @ weights) * leverage
    daily_interest_cost = interest_rate * (leverage - 1) / DAYS_PER_YEAR
    daily_port_returns_all -= daily_interest_cost

    # -------------------
    # Withdrawal strategies
    # -------------------
    strategies = {
        "Fixed 4%": fixed_pct_initial(initial_capital=START_CAPITAL, rate=0.04),
        "Variable 4%": pct_of_current_capital(rate=0.04),
        "Guardrail 2.5–5%": guardrail_pct(min_rate=0.025, max_rate=0.05),
        "Bucket Strategy": bucket_strategy(cash_years=3, annual_withdrawal=50_000)
    }

    all_results = {}
    xs = np.arange(N_DAYS + 1)

    # -------------------
    # Run simulation
    # -------------------
    for strat_name, strategy in strategies.items():
        wealth_paths = np.zeros((N_BOOTSTRAP, N_DAYS + 1))
        wealth_paths[:, 0] = START_CAPITAL

        for b in range(N_BOOTSTRAP):
            w = START_CAPITAL
            for t in range(N_DAYS):
                w = w * (1 + daily_port_returns_all[b, t])
                if t % DAYS_PER_YEAR == 0 and t > 0:
                    w -= strategy(w)
                wealth_paths[b, t + 1] = w

        final_wealth = wealth_paths[:, -1]
        terminated = (wealth_paths < 0).any(axis=1)

        all_results[strat_name] = {
            "wealth_paths": wealth_paths,
            "mean_path": wealth_paths.mean(axis=0),
            "std_path": wealth_paths.std(axis=0),
            "final_wealth": final_wealth,
            "terminated": terminated,
            "fail_rate": terminated.mean() * 100,
            "p16": np.percentile(final_wealth, 16),
            "p50": np.percentile(final_wealth, 50),
            "p84": np.percentile(final_wealth, 84),
            "prob_target": (final_wealth >= TARGET_WEALTH).mean() * 100,
        }

    # -------------------
    # Plot results
    # -------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True)
    axes = axes.flatten()
    for ax, (strat_name, res) in zip(axes, all_results.items()):
        wp = res["wealth_paths"]
        ax.fill_between(xs, 0, -START_CAPITAL*0.5, color='red', alpha=0.1)
        for i in range(min(50, N_BOOTSTRAP)):
            ax.plot(xs, np.where(wp[i] >= 0, wp[i], np.nan), color='blue', alpha=0.08)
        ax.plot(xs, res["mean_path"], color='black', linewidth=2)
        ax.fill_between(xs,
                        res["mean_path"] - res["std_path"],
                        res["mean_path"] + res["std_path"],
                        alpha=0.2)
        txt = (
            f"Fail rate: {res['fail_rate']:.1f}%\n"
            f"P16/P50/P84: ${res['p16']:,.0f}/${res['p50']:,.0f}/${res['p84']:,.0f}\n"
            f"Prob ≥ ${TARGET_WEALTH:,}: {res['prob_target']:.1f}%"
        )
        ax.annotate(txt, xy=(0.02, 0.95), xycoords='axes fraction',
                    va='top', fontsize=9, bbox=dict(boxstyle='round', alpha=0.1))
        ax.set_title(strat_name)
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
        ax.grid(alpha=0.3)

    plt.tight_layout()
    return all_results, fig
