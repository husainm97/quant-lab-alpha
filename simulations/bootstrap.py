"""
bootstrap.py

Generates 3 synthetic ETF return series, runs a block bootstrap simulation with a modular withdrawal strategy,
and plots portfolio wealth paths over a multi-year horizon with a translucent 1-sigma band around the mean.
Uses portfolio_module.Portfolio and withdrawal strategies from withdrawal_strategies.py.

IMPROVEMENTS:
- Input fields instead of sliders for stress tests
- Save/Load stress test configurations
- Configurable starting capital, withdrawal percentage, target wealth, etc.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import json
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

import statsmodels.api as sm

# Import fetch_ff5_monthly from parent directory
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data.fetch_ff5 import fetch_ff5_monthly


def save_config(config_dict, filename="simulation_config.json"):
    """Save simulation configuration to JSON file"""
    config_path = os.path.join(os.path.dirname(__file__), filename)
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    return config_path


def load_config(filename="simulation_config.json"):
    """Load simulation configuration from JSON file"""
    config_path = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return None


def run_simulation(root, portfolio_obj=None, returns_df=None, config=None):
    """
    Dashboard version: Math runs once, input fields update the view.
    Stats (P16/50/84) are in the Sidebar Card. Legend is on the Plot.
    """
    import tkinter as tk
    from tkinter import ttk, messagebox
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    if config is None: 
        config = {}

    # --- 1. DEFAULT SETTINGS ---
    MONTHS_PER_YEAR = 12
    N_BOOTSTRAP = config.get("N_BOOTSTRAP", 2000) 
    AVG_BLOCK_SIZE = config.get("BLOCK_SIZE", 60)
    
    # Load saved config if available
    saved_config = load_config()
    if saved_config:
        config.update(saved_config)
    
    START_CAPITAL = config.get("START_CAPITAL", 1_000_000)
    TARGET_WEALTH = config.get("TARGET_WEALTH", 2_700_000)
    TARGET_YEARS = config.get("TARGET_YEARS", 50)
    WITHDRAWAL_PCT = config.get("WITHDRAWAL_PCT", 4.0)
    N_MONTHS = TARGET_YEARS * MONTHS_PER_YEAR
    rng = np.random.default_rng(config.get("SEED", 42))

    if portfolio_obj is not None:
        returns_df = portfolio_obj.get_common_monthly_returns()
        tickers = list(returns_df.columns)
    else:
        raise ValueError("Provide a Portfolio object.")

    # --- FAMA-FRENCH EXTENSION ---
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
    for child in root.winfo_children(): 
        child.destroy()
    
    main_frame = ttk.Frame(root)
    main_frame.pack(fill="both", expand=True)

    # --- SIDEBAR ---
    sidebar = ttk.LabelFrame(main_frame, text="Simulation Controls", padding=10)
    sidebar.pack(side="left", fill="y", padx=5, pady=5)

    # === CONFIGURATION SECTION ===
    config_frame = ttk.LabelFrame(sidebar, text="Configuration", padding=5)
    config_frame.pack(fill="x", pady=5)

    # Starting Capital
    ttk.Label(config_frame, text="Starting Capital ($):").grid(row=0, column=0, sticky="w", pady=2)
    start_capital_var = tk.IntVar(value=START_CAPITAL)
    start_capital_entry = ttk.Entry(config_frame, textvariable=start_capital_var, width=15)
    start_capital_entry.grid(row=0, column=1, pady=2)

    # Target Wealth
    ttk.Label(config_frame, text="Target Wealth ($):").grid(row=1, column=0, sticky="w", pady=2)
    target_wealth_var = tk.IntVar(value=TARGET_WEALTH)
    target_wealth_entry = ttk.Entry(config_frame, textvariable=target_wealth_var, width=15)
    target_wealth_entry.grid(row=1, column=1, pady=2)

    # Target Years
    ttk.Label(config_frame, text="Target Years:").grid(row=2, column=0, sticky="w", pady=2)
    target_years_var = tk.IntVar(value=TARGET_YEARS)
    target_years_entry = ttk.Entry(config_frame, textvariable=target_years_var, width=15)
    target_years_entry.grid(row=2, column=1, pady=2)

    # Withdrawal Percentage
    ttk.Label(config_frame, text="Withdrawal % (Annual):").grid(row=3, column=0, sticky="w", pady=2)
    withdrawal_pct_var = tk.DoubleVar(value=WITHDRAWAL_PCT)
    withdrawal_pct_entry = ttk.Entry(config_frame, textvariable=withdrawal_pct_var, width=15)
    withdrawal_pct_entry.grid(row=3, column=1, pady=2)

    # === STRATEGY SECTION ===
    strategy_frame = ttk.LabelFrame(sidebar, text="Withdrawal Strategy", padding=5)
    strategy_frame.pack(fill="x", pady=5)

    strat_var = tk.StringVar(value="Fixed 4%")
    ttk.Label(strategy_frame, text="Strategy:").pack(anchor="w")
    strat_menu = ttk.Combobox(strategy_frame, textvariable=strat_var, state="readonly", 
                             values=["Fixed 4%", "Variable 4%", "Guardrail 2.5-5%", "Bucket Strategy"])
    strat_menu.pack(fill="x", pady=5)

    # === STRESS TEST SECTION (INPUT FIELDS) ===
    stress_frame = ttk.LabelFrame(sidebar, text="Stress Tests", padding=5)
    stress_frame.pack(fill="x", pady=5)

    # Return Shift
    ttk.Label(stress_frame, text="Return Shift (%/year):").grid(row=0, column=0, sticky="w", pady=2)
    ret_shift_var = tk.DoubleVar(value=0.0)
    ret_shift_entry = ttk.Entry(stress_frame, textvariable=ret_shift_var, width=10)
    ret_shift_entry.grid(row=0, column=1, pady=2, padx=5)
    ttk.Label(stress_frame, text="(-5 to +5)").grid(row=0, column=2, sticky="w")

    # Volatility Multiplier
    ttk.Label(stress_frame, text="Volatility Multiplier:").grid(row=1, column=0, sticky="w", pady=2)
    vol_mult_var = tk.DoubleVar(value=1.0)
    vol_mult_entry = ttk.Entry(stress_frame, textvariable=vol_mult_var, width=10)
    vol_mult_entry.grid(row=1, column=1, pady=2, padx=5)
    ttk.Label(stress_frame, text="(0.5 to 2.0)").grid(row=1, column=2, sticky="w")

    # === BUTTONS ===
    button_frame = ttk.Frame(sidebar)
    button_frame.pack(fill="x", pady=10)

    def save_current_config():
        """Save current configuration to file"""
        current_config = {
            "START_CAPITAL": start_capital_var.get(),
            "TARGET_WEALTH": target_wealth_var.get(),
            "TARGET_YEARS": target_years_var.get(),
            "WITHDRAWAL_PCT": withdrawal_pct_var.get(),
            "RETURN_SHIFT": ret_shift_var.get(),
            "VOL_MULT": vol_mult_var.get(),
            "STRATEGY": strat_var.get()
        }
        try:
            path = save_config(current_config)
            messagebox.showinfo("Success", f"Configuration saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration:\n{e}")

    def reset_to_defaults():
        """Reset all inputs to default values"""
        start_capital_var.set(1_000_000)
        target_wealth_var.set(2_700_000)
        target_years_var.set(50)
        withdrawal_pct_var.set(4.0)
        ret_shift_var.set(0.0)
        vol_mult_var.set(1.0)
        strat_var.set("Fixed 4%")
        update_plot()

    save_btn = ttk.Button(button_frame, text="💾 Save Config", command=save_current_config)
    save_btn.pack(side="left", padx=2, fill="x", expand=True)

    reset_btn = ttk.Button(button_frame, text="🔄 Reset", command=reset_to_defaults)
    reset_btn.pack(side="left", padx=2, fill="x", expand=True)

    update_btn = ttk.Button(button_frame, text="▶ Update Plot", command=lambda: update_plot())
    update_btn.pack(side="left", padx=2, fill="x", expand=True)

    # === STATS CARD ===
    stats_card = ttk.Label(sidebar, text="", font=("Courier", 10), justify="left", 
                           relief="sunken", padding=10, background="wheat")
    stats_card.pack(fill="both", expand=True, pady=10)

    # --- PLOT AREA ---
    fig, ax = plt.subplots(figsize=(10, 6))
    canvas = FigureCanvasTkAgg(fig, master=main_frame)
    canvas.get_tk_widget().pack(side="right", fill="both", expand=True)

    def update_plot(*args):
        """Update the plot with current parameters"""
        try:
            ax.clear()
            
            # Get current values
            current_start = start_capital_var.get()
            current_target = target_wealth_var.get()
            current_withdrawal = withdrawal_pct_var.get() / 100
            
            # Validate inputs
            if ret_shift_var.get() < -5 or ret_shift_var.get() > 5:
                messagebox.showwarning("Invalid Input", "Return shift must be between -5 and +5")
                ret_shift_var.set(max(-5, min(5, ret_shift_var.get())))
            
            if vol_mult_var.get() < 0.5 or vol_mult_var.get() > 2.0:
                messagebox.showwarning("Invalid Input", "Volatility multiplier must be between 0.5 and 2.0")
                vol_mult_var.set(max(0.5, min(2.0, vol_mult_var.get())))
            
            # Apply Stress
            r_shift = ret_shift_var.get() / 1200  # Annual to Monthly decimal
            v_mult = vol_mult_var.get()
            means = all_paths.mean(axis=1, keepdims=True)
            stressed_paths = (all_paths - means) * v_mult + means + r_shift
            
            # Portfolio Returns & Strategy
            port_rets = (stressed_paths @ weights) * leverage - monthly_interest
            
            # Select withdrawal strategy based on current settings
            if strat_var.get() == "Fixed 4%":
                strategy = fixed_pct_initial(current_start, current_withdrawal, scale=12)
            elif strat_var.get() == "Variable 4%":
                strategy = pct_of_current_capital(current_withdrawal, scale=12)
            elif strat_var.get() == "Guardrail 2.5-5%":
                strategy = guardrail_pct(0.025, 0.05, scale=12)
            else:  # Bucket Strategy
                strategy = bucket_strategy(cash_years=3, annual_withdrawal=current_start * current_withdrawal, scale=12)

            wealth_paths = np.zeros((N_BOOTSTRAP, N_MONTHS + 1))
            wealth_paths[:, 0] = current_start
            for b in range(N_BOOTSTRAP):
                w = current_start
                for t in range(N_MONTHS):
                    w *= (1 + port_rets[b, t])
                    if t > 0: 
                        w -= strategy(w)
                    if w <= 0: 
                        break
                    wealth_paths[b, t+1] = w

            # Visualization
            xs = np.arange(N_MONTHS + 1) / 12
            
            # Spaghetti lines (sample)
            for i in range(min(100, N_BOOTSTRAP)):
                path = wealth_paths[i]
                z_idx = np.where(path <= 0)[0]
                if len(z_idx) > 0:
                    ax.plot(xs[:z_idx[0]+1], path[:z_idx[0]+1], alpha=0.15, lw=0.7, color='blue')
                    ax.scatter(xs[z_idx[0]], 0, color='red', marker='x', s=30)
                else:
                    ax.plot(xs, path, alpha=0.15, lw=0.7, color='blue')

            # Bands & Median for Legend
            p5_path, p95_path = np.percentile(wealth_paths, [5, 95], axis=0)
            ax.plot(xs, np.median(wealth_paths, axis=0), color='darkgreen', alpha=0.8, lw=2.5, label="Median Path")
            ax.fill_between(xs, p5_path, p95_path, color='grey', alpha=0.15, label="5th - 95th Percentile")
            
            # Final Statistics for Summary Card
            fail_rate = (wealth_paths[:, -1] <= 0).mean() * 100
            p16, p50, p84 = np.percentile(wealth_paths[:, -1], [16, 50, 84])
            prob_target = (wealth_paths[:, -1] >= current_target).mean() * 100

            summary_txt = (
                f"--- SIMULATION RESULTS ---\n\n"
                f"Failure Rate: {fail_rate:>6.1f}%\n"
                f"P(≥ Target): {prob_target:>6.1f}%\n\n"
                f"--- FINAL WEALTH ---\n"
                f"P16: ${p16/1e6:>6.2f}M\n"
                f"P50: ${p50/1e6:>6.2f}M\n"
                f"P84: ${p84/1e6:>6.2f}M\n\n"
                f"--- CONFIG ---\n"
                f"Start: ${current_start/1e6:.2f}M\n"
                f"Target: ${current_target/1e6:.2f}M\n"
                f"Withdrawal: {current_withdrawal*100:.1f}%"
            )
            stats_card.config(text=summary_txt)

            # Plot Formatting
            ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
            padding = max(current_start, np.max(p95_path))*0.1
            ax.set_ylim(np.min(p5_path)-padding, np.max(p95_path)+padding)
            
            stress_label = ""
            if r_shift != 0 or v_mult != 1.0:
                stress_label = f" | Stress: Ret={ret_shift_var.get():+.1f}%, Vol={v_mult:.1f}x"
            
            ax.set_title(f"Monte Carlo: {strat_var.get()}{stress_label}", loc='center', fontweight='bold')
            ax.set_ylabel("Wealth (USD)")
            ax.set_xlabel("Years in Retirement")
            ax.legend(loc="upper left")
            ax.grid(alpha=0.3)
            canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error updating plot:\n{e}")

    # Bindings (only update on button click or Enter key)
    strat_menu.bind("<<ComboboxSelected>>", lambda e: update_plot())
    
    # Optional: Auto-update on Enter key in any entry field
    for entry in [start_capital_entry, target_wealth_entry, target_years_entry, 
                  withdrawal_pct_entry, ret_shift_entry, vol_mult_entry]:
        entry.bind("<Return>", lambda e: update_plot())
    
    # Initial plot
    update_plot()
    
    return None