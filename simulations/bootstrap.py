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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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

def run_simulation(root, portfolio_obj=None, returns_df=None, config=None):
    """
    Dashboard version: Math runs once, sliders update the view.
    Stats (P16/50/84) are in the Sidebar Card. Legend is on the Plot.
    """
    import tkinter as tk
    from tkinter import ttk
    import matplotlib.ticker as mtick
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    if config is None: config = {}

    # --- 1. SETTINGS & MATH (ORIGINAL LOGIC) ---
    MONTHS_PER_YEAR = 12
    N_BOOTSTRAP = config.get("N_BOOTSTRAP", 2000) 
    AVG_BLOCK_SIZE = config.get("BLOCK_SIZE", 60)
    START_CAPITAL = config.get("START_CAPITAL", 1_000_000)
    TARGET_WEALTH = config.get("TARGET_WEALTH", 2_700_000)
    TARGET_YEARS = config.get("TARGET_YEARS", 50)
    N_MONTHS = TARGET_YEARS * MONTHS_PER_YEAR
    rng = np.random.default_rng(config.get("SEED", 42))

    if portfolio_obj is not None:
        returns_df = portfolio_obj.get_common_monthly_returns()
        tickers = list(returns_df.columns)
    else:
        raise ValueError("Provide a Portfolio object.")

    # --- FAMA-FRENCH EXTENSION (FIXED NameError) ---
    ff_factors = fetch_ff5_monthly()
    ff_factors.index = pd.to_datetime(ff_factors.index).to_period('M').to_timestamp('M')
    factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    combined = pd.concat([returns_df, ff_factors], axis=1, join='inner')

    all_resids_dict = {t: (combined[t] - combined['RF']) - 
                      sm.OLS(combined[t] - combined['RF'], 
                             sm.add_constant(combined[factor_cols])).fit().predict(sm.add_constant(combined[factor_cols]))
                      for t in tickers}
    
    all_resids_df = pd.DataFrame(all_resids_dict)
    n_existing, n_needed = len(combined), max(0, N_MONTHS - len(combined))
    
    if n_needed > 0:
        sync_indices = []
        while len(sync_indices) < n_needed:
            block_len = rng.geometric(1 / AVG_BLOCK_SIZE)
            start = rng.integers(0, n_existing)
            end = min(start + block_len, n_existing)
            sync_indices.extend(range(start, end))
        
        sync_indices = sync_indices[:n_needed]
        synth_resids = all_resids_df.iloc[sync_indices].values
        ext_rets = {t: np.concatenate([combined[t].values, synth_resids[:, i] + combined['RF'].iloc[-1]]) 
                    for i, t in enumerate(tickers)}
        returns_extended_df = pd.DataFrame(ext_rets)
    else:
        returns_extended_df = combined[tickers]

    # --- PRE-COMPUTE BOOTSTRAP ---
    returns_matrix = returns_extended_df.values 
    weights = np.array([portfolio_obj.constituents[t] for t in tickers])
    leverage = portfolio_obj.leverage
    monthly_interest = portfolio_obj.interest_rate * (leverage - 1) / MONTHS_PER_YEAR

    all_paths = np.zeros((N_BOOTSTRAP, N_MONTHS, len(tickers)))
    for b in range(N_BOOTSTRAP):
        res = []
        while len(res) < N_MONTHS:
            start = rng.integers(0, len(returns_matrix))
            block_len = rng.geometric(1 / AVG_BLOCK_SIZE)
            end = min(start + block_len, len(returns_matrix))
            res.append(returns_matrix[start:end, :])
        all_paths[b] = np.vstack(res)[:N_MONTHS, :]

    # --- 2. GUI LAYOUT ---
    for child in root.winfo_children(): child.destroy()
    
    main_frame = ttk.Frame(root)
    main_frame.pack(fill="both", expand=True)

    sidebar = ttk.LabelFrame(main_frame, text="Simulation Controls", padding=10)
    sidebar.pack(side="left", fill="y", padx=5, pady=5)

    strat_var = tk.StringVar(value="Fixed 4%")
    ret_shift_var = tk.DoubleVar(value=0.0)
    vol_mult_var = tk.DoubleVar(value=1.0)

    ttk.Label(sidebar, text="Withdrawal Strategy:").pack(anchor="w")
    strat_menu = ttk.Combobox(sidebar, textvariable=strat_var, state="readonly", 
                             values=["Fixed 4%", "Variable 4%", "Guardrail 2.5-5%", "Bucket Strategy"])
    strat_menu.pack(fill="x", pady=5)

    ttk.Label(sidebar, text="Stress: Return Shift (%)").pack(anchor="w", pady=(10,0))
    ret_s = tk.Scale(sidebar, from_=-5.0, to=5.0, variable=ret_shift_var, orient="horizontal", resolution=0.1)
    ret_s.pack(fill="x")

    ttk.Label(sidebar, text="Stress: Volatility Multiplier").pack(anchor="w", pady=(10,0))
    vol_s = tk.Scale(sidebar, from_=0.5, to=2.0, variable=vol_mult_var, orient="horizontal", resolution=0.1)
    vol_s.pack(fill="x")

    # --- THE SUMMARY CARD (Sidebar Section) ---
    stats_card = ttk.Label(sidebar, text="", font=("Courier", 10), justify="left", 
                           relief="sunken", padding=10, background="wheat")
    stats_card.pack(fill="x", pady=20)

    # Plot area
    fig, ax = plt.subplots(figsize=(10, 6))
    canvas = FigureCanvasTkAgg(fig, master=main_frame)
    canvas.get_tk_widget().pack(side="right", fill="both", expand=True)

    def update_plot(*args):
        ax.clear()
        
        # Apply Stress
        r_shift = ret_shift_var.get() / 1200 # Annual to Monthly decimal
        v_mult = vol_mult_var.get()
        means = all_paths.mean(axis=1, keepdims=True)
        stressed_paths = (all_paths - means) * v_mult + means + r_shift
        
        # Portfolio Returns & Strategy
        port_rets = (stressed_paths @ weights) * leverage - monthly_interest
        strategy = {
            "Fixed 4%": fixed_pct_initial(START_CAPITAL, 0.04, scale=12),
            "Variable 4%": pct_of_current_capital(0.04, scale=12),
            "Guardrail 2.5-5%": guardrail_pct(0.025, 0.05, scale=12),
            "Bucket Strategy": bucket_strategy(cash_years=3, annual_withdrawal=50000, scale=12)
        }[strat_var.get()]

        wealth_paths = np.zeros((N_BOOTSTRAP, N_MONTHS + 1))
        wealth_paths[:, 0] = START_CAPITAL
        for b in range(N_BOOTSTRAP):
            w = START_CAPITAL
            for t in range(N_MONTHS):
                w *= (1 + port_rets[b, t])
                if t > 0: w -= strategy(w)
                if w <= 0: break
                wealth_paths[b, t+1] = w

        # Visualization
        xs = np.arange(N_MONTHS + 1) / 12
        
        # Spaghetti lines
        for i in range(min(100, N_BOOTSTRAP)):
            path = wealth_paths[i]
            z_idx = np.where(path <= 0)[0]
            if len(z_idx) > 0:
                ax.plot(xs[:z_idx[0]+1], path[:z_idx[0]+1], alpha=0.15, lw=0.7)
                ax.scatter(xs[z_idx[0]], 0, color='red', marker='x', s=30)
            else:
                ax.plot(xs, path, alpha=0.15, lw=0.7)

        # Bands & Median for Legend
        p5_path, p95_path = np.percentile(wealth_paths, [5, 95], axis=0)
        ax.plot(xs, np.median(wealth_paths, axis=0), color='grey', alpha=0.8, lw=2.5, label="Median Path")
        ax.fill_between(xs, p5_path, p95_path, color='grey', alpha=0.15, label="5th - 95th Percentile")
        
        # Final Statistics for Summary Card
        fail_rate = (wealth_paths[:, -1] <= 0).mean() * 100
        p16, p50, p84 = np.percentile(wealth_paths[:, -1], [16, 50, 84])
        prob_target = (wealth_paths[:, -1] >= TARGET_WEALTH).mean() * 100

        summary_txt = (
            f"--- RESULTS ---\n"
            f"Fail Rate: {fail_rate:>5.1f}%\n"
            f"P(Capital growth ≥ 2% p.a.): {prob_target:>4.1f}%\n\n"
            f"--- FINAL WEALTH ---\n"
            f"P16: ${p16/1e6:>5.2f}M\n"
            f"P50: ${p50/1e6:>5.2f}M\n"
            f"P84: ${p84/1e6:>5.2f}M"
        )
        stats_card.config(text=summary_txt)

        # Plot Formatting
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
        padding = max(START_CAPITAL, np.max(p95_path))*0.1
        ax.set_ylim(np.min(p5_path)-padding, np.max(p95_path)+padding)
        ax.set_title(f"Monte Carlo: {strat_var.get()}", loc='center', fontweight='bold')
        ax.set_ylabel("Wealth (USD)")
        ax.set_xlabel("Years in Retirement")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)
        canvas.draw()

    # Bindings
    strat_menu.bind("<<ComboboxSelected>>", lambda e: update_plot())
    ret_shift_var.trace_add("write", lambda *a: update_plot())
    vol_mult_var.trace_add("write", lambda *a: update_plot())
    
    update_plot()
    return None