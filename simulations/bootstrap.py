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

'''
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
    WITHDRAWAL_EVERY = 1  # Withdrawal every N months
    SEED = 42
    N_BOOTSTRAP = config.get("N_BOOTSTRAP", 2000)
    AVG_BLOCK_SIZE = config.get("BLOCK_SIZE", 60)  # average block size in months
    START_CAPITAL = config.get("START_CAPITAL", 1_000_000)
    TARGET_WEALTH = config.get("TARGET_WEALTH", 2_000_000)

    rng = np.random.default_rng(SEED)

    # ------------------------
    # Prepare returns
    # ------------------------
    if portfolio_obj is not None:
        returns_df = portfolio_obj.get_common_monthly_returns()
        tickers = list(returns_df.columns)
        N_ASSETS = len(tickers)
        N_MONTHS = returns_df.shape[0]
    else:
        raise ValueError("Provide a Portfolio object.")

    returns_matrix = returns_df.values  # shape (N_MONTHS, N_ASSETS)

    # ------------------------
    # Weights and leverage
    # ------------------------
    weights = np.array([portfolio_obj.constituents[t] for t in tickers])
    leverage = portfolio_obj.leverage
    interest_rate = portfolio_obj.interest_rate
    monthly_interest = interest_rate * (leverage - 1) / MONTHS_PER_YEAR

    # ------------------------
    # Stationary block bootstrap helper
    # ------------------------
    def block_bootstrap_multiasset(returns_matrix, avg_block_size, n_samples):
        N_months, N_assets = returns_matrix.shape
        result = []

        while len(result) < n_samples:
            start = rng.integers(0, N_months)
            block_len = rng.geometric(1/avg_block_size)
            end = min(start + block_len, N_months)
            result.append(returns_matrix[start:end, :])

        return np.vstack(result)[:n_samples, :]

    # ------------------------
    # Generate portfolio return paths
    # ------------------------
    all_paths = np.zeros((N_BOOTSTRAP, N_MONTHS, N_ASSETS))
    for b in range(N_BOOTSTRAP):
        all_paths[b] = block_bootstrap_multiasset(returns_matrix, AVG_BLOCK_SIZE, N_MONTHS)

    port_returns = (all_paths @ weights) * leverage - monthly_interest

    # ------------------------
    # Withdrawal strategies
    # ------------------------
    scale = MONTHS_PER_YEAR / WITHDRAWAL_EVERY
    strategies = {
        "Fixed 4%": fixed_pct_initial(START_CAPITAL, 0.04, scale=scale),
        "Variable 4%": pct_of_current_capital(0.04, scale=scale),
        "Guardrail 2.5-5%": guardrail_pct(0.025, 0.05, scale=scale),
        "Bucket Strategy": bucket_strategy(cash_years=3, annual_withdrawal=50_000, scale=scale)
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
                if t % WITHDRAWAL_EVERY == 0 and t > 0:
                    w -= strategy(w)
                wealth_paths[b, t + 1] = w

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
            ax.plot(xs, np.where(wp[i]>=0, wp[i], np.nan), alpha=0.08)
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
'''

import matplotlib.ticker as mtick
import statsmodels.api as sm
from data.fetch_ff5 import fetch_ff5_monthly

def run_simulation(portfolio_obj: Portfolio = None, returns_df: pd.DataFrame = None, config: dict = None):
    """
    Run a block-bootstrap Monte Carlo simulation using monthly returns,
    extending short histories with Fama-French 5-factor fits.
    """
    if config is None:
        config = {}

    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    MONTHS_PER_YEAR = 12
    WITHDRAWAL_EVERY = 1
    SEED = config.get("SEED", 42)
    N_BOOTSTRAP = config.get("N_BOOTSTRAP", 5000)
    AVG_BLOCK_SIZE = config.get("BLOCK_SIZE", 60)
    START_CAPITAL = config.get("START_CAPITAL", 1_000_000)
    TARGET_WEALTH = config.get("TARGET_WEALTH", 2_000_000)
    TARGET_YEARS = config.get("TARGET_YEARS", 50)

    rng = np.random.default_rng()

    if portfolio_obj is not None:
        returns_df = portfolio_obj.get_common_monthly_returns()
        tickers = list(returns_df.columns)
    else:
        raise ValueError("Provide a Portfolio object.")

    ff_factors = fetch_ff5_monthly()
    ff_factors.index = pd.to_datetime(ff_factors.index).to_period('M').to_timestamp('M')
    factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']

    combined = pd.concat([returns_df, ff_factors], axis=1, join='inner')

    synthetic_note_added = False
    extended_returns = []
    N_SYNTH = TARGET_YEARS * MONTHS_PER_YEAR

    for etf in tickers:
        y = combined[etf] - combined['RF']
        X = sm.add_constant(combined[factor_cols])
        model = sm.OLS(y, X).fit()
        residuals = y - model.predict(X)

        n_needed = max(0, N_SYNTH - len(y))
        if n_needed > 0:
            synthetic_note_added = True
            blocks = []
            while sum(len(b) for b in blocks) < n_needed:
                block_len = rng.geometric(1 / AVG_BLOCK_SIZE)
                start = rng.integers(0, len(residuals))
                end = min(start + block_len, len(residuals))
                blocks.append(residuals.iloc[start:end].values)
            synthetic_resid = np.concatenate(blocks)[:n_needed]

            synthetic_index = pd.date_range(
                end=combined.index[-1] + pd.DateOffset(months=n_needed),
                periods=n_needed, freq='ME'
            )
            extended_series = pd.concat([
                y + combined['RF'],
                pd.Series(synthetic_resid + combined['RF'].iloc[-1], index=synthetic_index)
            ])
        else:
            extended_series = y + combined['RF']

        extended_returns.append(extended_series)

    returns_extended_df = pd.DataFrame({t: s.values for t, s in zip(tickers, extended_returns)})
    returns_extended_df.index = pd.date_range(start=returns_extended_df.index[0],
                                              periods=len(returns_extended_df), freq='ME')

    N_ASSETS = len(tickers)
    N_MONTHS = returns_extended_df.shape[0]
    returns_matrix = returns_extended_df.values

    weights = np.array([portfolio_obj.constituents[t] for t in tickers])
    leverage = portfolio_obj.leverage
    interest_rate = portfolio_obj.interest_rate
    monthly_interest = interest_rate * (leverage - 1) / MONTHS_PER_YEAR

    def block_bootstrap_multiasset(returns_matrix, avg_block_size, n_samples):
        N_months, N_assets = returns_matrix.shape
        result = []
        while len(result) < n_samples:
            start = rng.integers(0, N_months)
            block_len = rng.geometric(1 / avg_block_size)
            end = min(start + block_len, N_months)
            block = returns_matrix[start:end, :]
            result.append(block)
        return np.vstack(result)[:n_samples, :]

    all_paths = np.zeros((N_BOOTSTRAP, N_MONTHS, N_ASSETS))
    for b in range(N_BOOTSTRAP):
        all_paths[b] = block_bootstrap_multiasset(returns_matrix, AVG_BLOCK_SIZE, N_MONTHS)

    port_returns = (all_paths @ weights) * leverage - monthly_interest

    scale = MONTHS_PER_YEAR / WITHDRAWAL_EVERY
    strategies = {
        "Fixed 4%": fixed_pct_initial(START_CAPITAL, 0.04, scale=scale),
        "Variable 4%": pct_of_current_capital(0.04, scale=scale),
        "Guardrail 2.5-5%": guardrail_pct(0.025, 0.05, scale=scale),
        "Bucket Strategy": bucket_strategy(cash_years=3, annual_withdrawal=50_000, scale=scale)
    }

    all_results = {}
    xs = np.arange(N_MONTHS + 1) / 12  # months → years

    for name, strategy in strategies.items():
        wealth_paths = np.zeros((N_BOOTSTRAP, N_MONTHS + 1))
        wealth_paths[:, 0] = START_CAPITAL
        for b in range(N_BOOTSTRAP):
            w = START_CAPITAL
            for t in range(N_MONTHS):
                w *= 1 + port_returns[b, t]
                if t % WITHDRAWAL_EVERY == 0 and t > 0:
                    w -= strategy(w)
                wealth_paths[b, t + 1] = w

        final_wealth = wealth_paths[:, -1]
        terminated = (wealth_paths < 0).any(axis=1)

        all_results[name] = {
            "wealth_paths": wealth_paths,
            "mean_path": wealth_paths.mean(axis=0),
            "std_path": wealth_paths.std(axis=0),
            "final_wealth": final_wealth,
            "terminated": terminated,
            "fail_rate": terminated.mean() * 100,
            "p16": np.percentile(final_wealth, 16),
            "p50": np.percentile(final_wealth, 50),
            "p84": np.percentile(final_wealth, 84),
            "prob_target": (final_wealth >= TARGET_WEALTH).mean() * 100
        }

    # Determine consistent y-limits across all strategies
    all_min = min(res["wealth_paths"].min() for res in all_results.values())
    all_max = max(res["wealth_paths"].max() for res in all_results.values())

    n_strats = len(strategies)
    n_cols = 2
    n_rows = int(np.ceil(n_strats / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 12), sharex=True, sharey=True)
    axes = axes.flatten()

    note_text = "Note: Extra history was synthesised using a Fama-French 5 Factor Regression." \
        if synthetic_note_added else ""

    for ax, (name, res) in zip(axes, all_results.items()):
        wp = res["wealth_paths"]

        # Danger zone: anything below zero
        ax.fill_between(xs, -np.inf, 0, color='red', alpha=0.1)

        # Sample paths
        for i in range(min(100, N_BOOTSTRAP)):
            path = wp[i]
            ax.plot(xs, np.where(path >= 0, path, np.nan), alpha=0.15)
            ax.plot(xs, np.where(path < 0, path, np.nan), '--', color='red', alpha=0.5)

        # Mean path in transparent grey with ±1 std band
        ax.plot(xs, res["mean_path"], color='grey', alpha=0.7, linewidth=2)
        ax.fill_between(xs,
                        res["mean_path"] - res["std_path"],
                        res["mean_path"] + res["std_path"],
                        color='grey', alpha=0.2)

        # Improved annotation
        txt = (
            f"Fail rate: {res['fail_rate']:.1f}%\n"
            f"P16 / P50 / P84: ${res['p16']:,.0f} / ${res['p50']:,.0f} / ${res['p84']:,.0f}\n"
            f"Prob ≥ ${TARGET_WEALTH:,}: {res['prob_target']:.1f}%\n"
            f"{note_text}"
        )
        ax.annotate(txt, xy=(0.02, 0.95), xycoords='axes fraction', va='top', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', alpha=0.15, facecolor='white'))

        ax.set_title(name, fontsize=12)
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
        ax.grid(alpha=0.3)

    # Hide unused axes
    for ax in axes[n_strats:]:
        ax.axis('off')

    axes[-1].set_xlabel("Years", fontsize=12)
    plt.tight_layout()

    return all_results, fig



