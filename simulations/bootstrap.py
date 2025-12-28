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

    MONTHS_PER_YEAR = 12
    WITHDRAWAL_EVERY = 1
    SEED = config.get("SEED", 42)
    N_BOOTSTRAP = config.get("N_BOOTSTRAP", 10000)
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
    N_SYNTH = TARGET_YEARS * MONTHS_PER_YEAR
    
    # --- PRESERVE CORRELATION: SYNCHRONIZED RESIDUAL SAMPLING ---
    # 1. Calculate all residuals first
    all_resids_dict = {}
    for etf in tickers:
        y = combined[etf] - combined['RF']
        X = sm.add_constant(combined[factor_cols])
        model = sm.OLS(y, X).fit()
        all_resids_dict[etf] = y - model.predict(X)
    
    all_resids_df = pd.DataFrame(all_resids_dict)
    n_existing = len(combined)
    n_needed = max(0, N_SYNTH - n_existing)
    
    if n_needed > 0:
        synthetic_note_added = True
        # Draw synchronized indices for all assets
        sync_indices = []
        while len(sync_indices) < n_needed:
            block_len = rng.geometric(1 / AVG_BLOCK_SIZE)
            start = rng.integers(0, n_existing)
            end = min(start + block_len, n_existing)
            sync_indices.extend(range(start, end))
        
        sync_indices = sync_indices[:n_needed]
        synthetic_resids_matrix = all_resids_df.iloc[sync_indices].values
        
        synthetic_index = pd.date_range(
            end=combined.index[-1] + pd.DateOffset(months=n_needed),
            periods=n_needed, freq='ME'
        )
        
        # Reconstruct extended returns
        extended_returns = []
        for i, etf in enumerate(tickers):
            hist_y = combined[etf].values
            # Re-add RF (using last known RF for synthetic portion)
            synth_y = synthetic_resids_matrix[:, i] + combined['RF'].iloc[-1]
            extended_series = pd.Series(np.concatenate([hist_y, synth_y]))
            extended_returns.append(extended_series)
    else:
        extended_returns = [combined[etf] for etf in tickers]

    returns_extended_df = pd.DataFrame({t: s.values for t, s in zip(tickers, extended_returns)})
    returns_extended_df.index = pd.date_range(start=combined.index[0],
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
    xs = np.arange(N_MONTHS + 1) / 12

    for name, strategy in strategies.items():
        wealth_paths = np.zeros((N_BOOTSTRAP, N_MONTHS + 1))
        wealth_paths[:, 0] = START_CAPITAL
        for b in range(N_BOOTSTRAP):
            w = START_CAPITAL
            for t in range(N_MONTHS):
                w *= 1 + port_returns[b, t]
                if t % WITHDRAWAL_EVERY == 0 and t > 0:
                    w -= strategy(w)
                
                # --- PATH TERMINATION ---
                if w <= 0:
                    wealth_paths[b, t + 1] = 0
                    break
                wealth_paths[b, t + 1] = w

        final_wealth = wealth_paths[:, -1]
        terminated = (wealth_paths <= 0).any(axis=1)

        all_results[name] = {
            "wealth_paths": wealth_paths,
            "mean_path": wealth_paths.mean(axis=0),
            "p5_path": np.percentile(wealth_paths, 5, axis=0),   # Requested Percentiles
            "p95_path": np.percentile(wealth_paths, 95, axis=0),
            "final_wealth": final_wealth,
            "terminated": terminated,
            "fail_rate": terminated.mean() * 100,
            "p16": np.percentile(final_wealth, 16),
            "p50": np.percentile(final_wealth, 50),
            "p84": np.percentile(final_wealth, 84),
            "prob_target": (final_wealth >= TARGET_WEALTH).mean() * 100
        }

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

        ax.fill_between(xs, -np.inf, 0, color='red', alpha=0.1)

        # --- TERMINATED PATH PLOTTING ---
        for i in range(min(100, N_BOOTSTRAP)):
            path = wp[i]
            # Find index where path hits zero
            zero_idx = np.where(path <= 0)[0]
            if len(zero_idx) > 0:
                ax.plot(xs[:zero_idx[0]+1], path[:zero_idx[0]+1], alpha=0.15)
                ax.scatter(xs[zero_idx[0]], 0, color='red', marker='x', s=40, zorder=5)
            else:
                ax.plot(xs, path, alpha=0.15)

        # Mean path and Percentile 5-95 bands
        ax.plot(xs, res["mean_path"], color='grey', alpha=0.7, linewidth=2)
        ax.fill_between(xs, res["p5_path"], res["p95_path"], color='grey', alpha=0.2)

        txt = (
            f"Fail rate: {res['fail_rate']:.1f}%\n"
            f"P16 / P50 / P84: ${res['p16']:,.0f} / ${res['p50']:,.0f} / ${res['p84']:,.0f}\n"
            f"Prob ≥ ${TARGET_WEALTH:,}: {res['prob_target']:.1f}%\n"
            f"{note_text}"
        )
        ax.annotate(txt, xy=(0.02, 0.95), xycoords='axes fraction', va='top', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', alpha=0.5, facecolor='wheat'))

        ax.set_title(name, fontsize=12)
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
        ax.set_ylim(bottom=-100000) # Slightly below zero to see the Xs
        ax.grid(alpha=0.3)

    for ax in axes[n_strats:]:
        ax.axis('off')

    axes[-1].set_xlabel("Years", fontsize=12)
    plt.tight_layout()

    return all_results, fig