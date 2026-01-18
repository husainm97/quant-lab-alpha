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
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
import yfinance as yf
import matplotlib

# Ensure local imports work
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from portfolio_module import Portfolio
from withdrawal_strategies import (
    fixed_pct_initial,
    pct_of_current_capital,
    guardrail_pct,
    bucket_strategy,
    inflation_adjusted
)

import matplotlib.ticker as mtick
import statsmodels.api as sm
from data.fetch_ff5 import fetch_ff5_monthly


def create_simulation_settings(parent, base_currency):
    frame = ttk.LabelFrame(parent, text="Simulation Settings", padding=5)
    
    # Example numeric fields
    start_capital_var = tk.DoubleVar(value=1_000_000)
    target_wealth_var = tk.DoubleVar(value=2_430_000)
    target_years_var = tk.IntVar(value=30)
    n_bootstrap_var = tk.IntVar(value=10000)
    
    for text, var_ in [(f"Start Capital ({base_currency})", start_capital_var),
                        (f"Target Wealth ({base_currency})", target_wealth_var),
                        (f"Target Years", target_years_var),
                        (f"Bootstrap Samples", n_bootstrap_var)]:
        ttk.Label(frame, text=text).pack(anchor="w")
        ttk.Entry(frame, textvariable=var_).pack(fill="x", pady=2)
    
    return frame, start_capital_var, target_wealth_var, target_years_var, n_bootstrap_var

def create_strategy_config(parent):
    frame = ttk.LabelFrame(parent, text="Strategy Settings", padding=5)
    
    strat_var = tk.StringVar(value="Fixed % Initial")
    ttk.Label(frame, text="Strategy:").pack(anchor="w")
    strat_menu = ttk.Combobox(frame, textvariable=strat_var, state="readonly",
                              values=["Fixed % Initial", "Fixed % Current", "Guardrail 2.5-5%", "Bucket Strategy"])
    strat_menu.pack(fill="x", pady=2)
    
    # Container for dynamic fields
    dyn_frame = ttk.Frame(frame)
    dyn_frame.pack(fill="x", pady=5)
    
    # Dynamic fields example
    lower_var = tk.DoubleVar(value=2.5)
    upper_var = tk.DoubleVar(value=4.0)
    annual_withdrawal_var = tk.DoubleVar(value=50_000)
    cash_years_var = tk.IntVar(value=3)
    
    def update_dynamic_fields(*args):
        for widget in dyn_frame.winfo_children():
            widget.destroy()
        strat = strat_var.get()
        if "Guardrail" in strat:
            ttk.Label(dyn_frame, text="Lower Rail:").pack(anchor="w")
            ttk.Entry(dyn_frame, textvariable=lower_var).pack(fill="x")
            ttk.Label(dyn_frame, text="Upper Rail:").pack(anchor="w")
            ttk.Entry(dyn_frame, textvariable=upper_var).pack(fill="x")
        elif "Bucket" in strat:
            ttk.Label(dyn_frame, text="Cash Years:").pack(anchor="w")
            ttk.Entry(dyn_frame, textvariable=cash_years_var).pack(fill="x")
            ttk.Label(dyn_frame, text="Annual Withdrawal:").pack(anchor="w")
            ttk.Entry(dyn_frame, textvariable=annual_withdrawal_var).pack(fill="x")
        elif "Fixed % Current" in strat:
            ttk.Label(dyn_frame, text="Withdrawal %:").pack(anchor="w")
            ttk.Entry(dyn_frame, textvariable=upper_var).pack(fill="x")  # reuse upper_var for % example
        elif "Fixed % Initial" in strat:
            ttk.Label(dyn_frame, text="Withdrawal %:").pack(anchor="w")
            ttk.Entry(dyn_frame, textvariable=upper_var).pack(fill="x")
        # Fixed 4% has no extra fields
    strat_var.trace_add("write", update_dynamic_fields)
    update_dynamic_fields()
    
    return frame, strat_var, lower_var, upper_var, cash_years_var, annual_withdrawal_var

# -------------------------
# 4. Stats Card with Theme
# -------------------------
def create_stats_card(parent, colors):
    card = ttk.Label(parent, text="Simulation Stats", justify="left", padding=10)
    card.configure(background=colors["card_bg"], foreground=colors["fg"])
    card.pack(fill="x", pady=10)
    return card

def get_theme_colors(is_dark=False):
    if is_dark:
        plt.style.use('dark_background')
        return {
            "bg": "#222222",
            "fg": "white",
            "line": "lightgrey",
            "envelope": "lightgrey",
            "fail_marker": "red",
            "card_bg": "#333333"
        }
    else:
        plt.style.use('default')
        return {
            "bg": "white",
            "fg": "black",
            "line": "grey",
            "envelope": "lightgrey",
            "fail_marker": "red",
            "card_bg": "#CECECE"
        }


def create_slider_with_entry(parent, label_text, var, from_, to_, resolution=0.1):
    frame = ttk.Frame(parent)
    ttk.Label(frame, text=label_text).pack(anchor="w")
    
    slider = tk.Scale(frame, from_=from_, to=to_, orient="horizontal", resolution=resolution, variable=var)
    slider.pack(fill="x")
    
    entry_var = tk.StringVar(value=f"{var.get():.2f}")
    entry = ttk.Entry(frame, textvariable=entry_var, width=6)
    entry.pack(anchor="e", pady=2)
    
    # Keep slider and entry in sync
    def update_entry(*args):
        entry_var.set(f"{var.get():.2f}")
    def update_slider(*args):
        try:
            val = float(entry_var.get())
            val = max(min(val, to_), from_)  # clamp
            var.set(val)
        except ValueError:
            pass
    var.trace_add("write", update_entry)
    entry_var.trace_add("write", update_slider)
    
    return frame


def run_simulation(root, portfolio_obj=None, returns_df=None, config=None, is_dark=False):
    """
    Dashboard version: Math runs once, sliders update the view.
    Stats (P16/50/84) are in the Sidebar Card. Legend is on the Plot.
    """
    import tkinter as tk
    from tkinter import ttk
    import matplotlib.ticker as mtick
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    if config is None:
        config = {}
    if portfolio_obj is None:
        raise ValueError("Provide a Portfolio object.")

    # --- CONSTANTS ---
    MONTHS_PER_YEAR = 12
    INFLATION_RATE = config.get("INFLATION_RATE", 0.03)
    AVG_BLOCK_SIZE = config.get("BLOCK_SIZE", 60)
    SEED = config.get("SEED", None) # Set to 42 for testing
    colors = get_theme_colors(is_dark)

    # --- CLEAR ROOT AND SETUP MAIN FRAME ---
    for child in root.winfo_children():
        child.destroy()
    
    main_frame = ttk.Frame(root)
    main_frame.pack(fill="both", expand=True)

    # --- SIDEBAR SETUP ---
    sidebar = ttk.LabelFrame(main_frame, text="Simulation Controls", padding=10)
    sidebar.pack(side="left", fill="y", padx=5, pady=5)

    # Simulation settings widgets
    sim_frame, start_capital_var, target_wealth_var, target_years_var, n_bootstrap_var = create_simulation_settings(sidebar, portfolio_obj.base_currency)
    sim_frame.pack(fill="x", pady=5)

    # Strategy settings widgets
    strat_frame, strat_var, lower_var, upper_var, cash_years_var, annual_withdrawal_var = create_strategy_config(sidebar)
    strat_frame.pack(fill="x", pady=5)
    strat_menu = strat_frame.winfo_children()[1]  # Combobox

    # Stress test sliders
    ret_shift_var = tk.DoubleVar(value=0.0)
    vol_mult_var = tk.DoubleVar(value=1.0)
    inflation_var = tk.DoubleVar(value=INFLATION_RATE * 100)

    ret_shift_frame = create_slider_with_entry(sidebar, "Stress: Return Shift (%)", ret_shift_var, -5.0, 5.0, resolution=0.1)
    ret_shift_frame.pack(fill="x", pady=5)

    vol_mult_frame = create_slider_with_entry(sidebar, "Stress: Volatility Multiplier", vol_mult_var, 0.5, 2.0, resolution=0.1)
    vol_mult_frame.pack(fill="x", pady=5)

    inflation_frame = create_slider_with_entry(sidebar, "Inflation Adjustment (%)", inflation_var, 0.0, 10.0, resolution=0.1)
    inflation_frame.pack(fill="x", pady=5)
    inflation_slider = inflation_frame.winfo_children()[1]

    # Stats card
    stats_card = ttk.Label(sidebar, text="", font=("Courier", 10), justify="left", 
                          relief="sunken", padding=10, background=colors["card_bg"], 
                          foreground=colors["fg"])
    stats_card.pack(fill="x", pady=20)

    # View selector
    view_var = tk.StringVar(value="Simulation")
    view_frame = ttk.LabelFrame(sidebar, text="Display Mode", padding=5)
    view_frame.pack(fill="x", pady=5)
    ttk.Radiobutton(view_frame, text="Monte Carlo", variable=view_var, 
                    value="Simulation", command=lambda: update_plot()).pack(side="left", expand=True)
    ttk.Radiobutton(view_frame, text="Diagnostic", variable=view_var, 
                    value="Diagnostic", command=lambda: update_plot()).pack(side="left", expand=True)

    # --- PLOT AREA ---
    fig = matplotlib.figure.Figure(figsize=(10, 6))
    canvas = FigureCanvasTkAgg(fig, master=main_frame)
    canvas.get_tk_widget().pack(side="right", fill="both", expand=True)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    toolbar.pack(fill="x")

    # --- SHARED STATE ---
    update_job = None
    simulation_data = {
        'all_paths': None,
        'diagnostic_data': None,
        'N_MONTHS': None
    }

    # --- PREPARE BASE DATA (ONE-TIME) ---
    returns_df = portfolio_obj.get_common_monthly_returns(use_usd=True)
    tickers = list(returns_df.columns)
    weights = np.array([portfolio_obj.constituents[t] for t in tickers])
    leverage = portfolio_obj.leverage
    monthly_interest = portfolio_obj.interest_rate * (leverage - 1) / MONTHS_PER_YEAR

    # Fetch Fama-French factors
    ff_factors = fetch_ff5_monthly(region=portfolio_obj.factor_region)
    ff_factors.index = pd.to_datetime(ff_factors.index).to_period('M').to_timestamp('M')
    factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    combined = pd.concat([returns_df, ff_factors], axis=1, join='inner')

    # Compute residuals for each ticker
    all_resids_dict = {
        t: (combined[t] - combined['RF']) - 
           sm.OLS(combined[t] - combined['RF'], 
                  sm.add_constant(combined[factor_cols])).fit().predict(sm.add_constant(combined[factor_cols]))
        for t in tickers
    }
    all_resids_df = pd.DataFrame(all_resids_dict)

    # Compute beta matrix for factor fit
    beta_matrix = np.array([
        sm.OLS(combined[t] - combined['RF'], sm.add_constant(combined[factor_cols])).fit().params.values
        for t in tickers
    ]).T  # shape: (1 + n_factors, n_assets)

    # FX conversion setup
    fx_returns_hist = None
    if portfolio_obj.base_currency != "USD":
        fx_ticker = f"USD{portfolio_obj.base_currency}=X"
        fx_df = yf.download(fx_ticker, start=returns_df.index[0], end=returns_df.index[-1], 
                           progress=False, auto_adjust=True)
        if isinstance(fx_df.columns, pd.MultiIndex):
            fx_df.columns = [col[0] for col in fx_df.columns]
        fx_monthly = fx_df['Close'].resample('ME').last().pct_change().fillna(0)
        fx_returns_hist = fx_monthly.reindex(combined.index, method='ffill').fillna(0).values

    # --- HELPER: EXTEND RETURNS DATA ---
    def extend_returns_data(N_MONTHS):
        """
        Generate extended returns matrix deterministically. 
        Uses a one-time fit of factor data to extend history.
        """
        n_existing = len(combined)
        n_needed = max(0, N_MONTHS - n_existing)
        
        factor_returns = combined[factor_cols].values
        
        if n_needed > 0:
            # 1. Deterministic Factor Extension
            # Instead of bootstrap, we tile the factor returns to reach n_needed
            # This ensures if you run it 10M times, the 'extended' factor set is identical
            indices = np.arange(n_needed) % n_existing
            synth_factors = factor_returns[indices]
            
            # 2. Use the Model to Extend (Predict returns without residuals)
            # synth_factor_pred shape: (n_needed, n_assets)
            synth_factor_pred = sm.add_constant(synth_factors) @ beta_matrix

            # 3. Construct Extended Returns
            # Synthetic (Model Fit) followed by Historical (Actual Data)
            ext_rets = {
                t: np.concatenate([synth_factor_pred[:, i], combined[t].values])
                for i, t in enumerate(tickers)
            }
            returns_extended_df = pd.DataFrame(ext_rets)

            # --- Diagnostic data setup ---
            hist_port_returns = combined[tickers].values @ weights
            synth_port_returns = synth_factor_pred @ weights
            
            # Create a deterministic index for the synthetic past
            synth_index = pd.date_range(end=combined.index[0] - pd.offsets.MonthEnd(1), 
                                        periods=n_needed, freq='ME')
            combined_index = synth_index.append(combined.index)
            combined_returns = np.concatenate([synth_port_returns, hist_port_returns])
            combined_prices = 100 * np.cumprod(1 + combined_returns)
            
            # Factor fit for visualization on the historical part
            X_factors_hist = sm.add_constant(combined[factor_cols])
            port_excess = hist_port_returns - combined['RF'].values
            port_model = sm.OLS(port_excess, X_factors_hist).fit()
            port_fit_returns = port_model.predict(X_factors_hist) + combined['RF'].values
            
            price_at_hist_start = combined_prices[n_needed - 1]
            port_fit_prices = price_at_hist_start * np.cumprod(1 + port_fit_returns)

            diagnostic_data = {
                'combined_index': combined_index,
                'combined_prices': combined_prices,
                'hist_index': combined.index,
                'port_fit_prices': port_fit_prices,
                'port_fit_returns': port_fit_returns,
                'hist_port_returns': hist_port_returns,
                'synth_index': synth_index,
                'synth_port_returns': synth_port_returns,
                'n_needed': n_needed
            }
        else:
            # Case where existing history is long enough
            returns_extended_df = combined[tickers]
            hist_port_returns = combined[tickers].values @ weights
            X_factors = sm.add_constant(combined[factor_cols])
            port_excess = hist_port_returns - combined['RF'].values
            port_model = sm.OLS(port_excess, X_factors).fit()
            port_fit_returns = port_model.predict(X_factors) + combined['RF'].values
            
            combined_prices = 100 * np.cumprod(1 + hist_port_returns)
            port_fit_prices = 100 * np.cumprod(1 + port_fit_returns)

            diagnostic_data = {
                'combined_index': combined.index,
                'combined_prices': combined_prices,
                'hist_index': combined.index,
                'port_fit_prices': port_fit_prices,
                'port_fit_returns': port_fit_returns,
                'hist_port_returns': hist_port_returns,
                'n_needed': 0
            }

        # Apply FX conversion if needed (Deterministic tiling)
        returns_matrix = returns_extended_df.values
        if portfolio_obj.base_currency != "USD":
            if n_needed > 0:
                fx_indices = np.arange(len(returns_matrix)) % len(fx_returns_hist)
                fx_extended = fx_returns_hist[fx_indices]
            else:
                fx_extended = fx_returns_hist[:N_MONTHS]
            returns_matrix = (1 + returns_matrix) * (1 + fx_extended[:, np.newaxis]) - 1

        return returns_matrix, diagnostic_data

    # --- CORE: RECOMPUTE SIMULATION ---
    def recompute_simulation():
        """Regenerate bootstrap paths based on current settings."""
        N_MONTHS = target_years_var.get() * MONTHS_PER_YEAR
        n_bootstrap = n_bootstrap_var.get()
        
        simulation_data['N_MONTHS'] = N_MONTHS

        # Extend returns to match target length
        returns_matrix, diagnostic_data = extend_returns_data(N_MONTHS)
        simulation_data['diagnostic_data'] = diagnostic_data

        # Progress window
        progress_win = tk.Toplevel(root)
        progress_win.title("Computing Simulation")
        progress_win.geometry("400x120")
        progress_win.transient(root)
        progress_win.grab_set()
        
        ttk.Label(progress_win, text="Generating bootstrap paths...", font=("Arial", 10)).pack(pady=10)
        progress_bar = ttk.Progressbar(progress_win, length=350, mode='determinate', maximum=n_bootstrap)
        progress_bar.pack(pady=10)
        status_label = ttk.Label(progress_win, text="", font=("Arial", 9))
        status_label.pack()
        progress_win.update()

        # Generate bootstrap paths
        rng = np.random.default_rng(SEED)
        all_paths = np.zeros((n_bootstrap, N_MONTHS, len(tickers)))
        
        for b in range(n_bootstrap):
            res = []
            while len(res) < N_MONTHS:
                start = rng.integers(0, len(returns_matrix))
                block_len = rng.geometric(1 / AVG_BLOCK_SIZE)
                end = min(start + block_len, len(returns_matrix))
                res.append(returns_matrix[start:end, :])
            all_paths[b] = np.vstack(res)[:N_MONTHS, :]
            
            if b % max(1, n_bootstrap // 100) == 0:
                progress_bar['value'] = b + 1
                status_label.config(text=f"Path {b+1}/{n_bootstrap}")
                progress_win.update()
        
        progress_win.destroy()
        simulation_data['all_paths'] = all_paths

    # --- PLOTTING FUNCTION ---
    def update_plot(*args):
        """Update the plot based on current view and slider values."""
        fig.clear()

        if view_var.get() == "Diagnostic":
            # --- DIAGNOSTIC VIEW ---
            diag = simulation_data['diagnostic_data']
            if diag is None:
                return

            gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1], sharex=ax1)

            n_needed = diag['n_needed']
            
            if n_needed > 0:
                ax1.plot(diag['combined_index'][:n_needed+1], diag['combined_prices'][:n_needed+1], 
                        label="Synthetic Pre-History", color='orange', lw=1.5, alpha=0.8)
                ax2.plot(diag['synth_index'], diag['synth_port_returns'], color='orange', alpha=0.4)

            ax1.plot(diag['hist_index'], diag['combined_prices'][max(0, n_needed):], 
                    label="Historic Price Data", color='#1f77b4', lw=2.5)
            ax1.plot(diag['hist_index'], diag['port_fit_prices'], 
                    label="Fama-French 5-Factor Fit", color='red', linestyle="--", lw=1.5)

            ax1.set_ylabel("Portfolio Value (USD)")
            ax1.set_title("Pre-History Extension & Factor Alignment", fontweight='bold')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
            ax1.yaxis.set_major_formatter(mtick.StrMethodFormatter('USD {x:,.0f}'))

            ax2.plot(diag['hist_index'], diag['hist_port_returns'], 
                    label="Actual Returns", color='#1f77b4', alpha=0.5)
            ax2.plot(diag['hist_index'], diag['port_fit_returns'], 
                    label="Fitted Returns", color='red', linestyle="--", lw=1, alpha=0.7)
            ax2.set_ylabel("Monthly Returns")
            ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='lower left', fontsize='small', ncol=2)

        else:
            # --- SIMULATION VIEW ---
            if simulation_data['all_paths'] is None:
                return

            ax = fig.add_subplot(111)
            
            # Get current slider values
            r_shift = ret_shift_var.get() / 1200
            v_mult = vol_mult_var.get()
            inflation = inflation_var.get() / 100.0
            
            # Apply stress to paths
            all_paths = simulation_data['all_paths']
            means = all_paths.mean(axis=1, keepdims=True)
            stressed_paths = (all_paths - means) * v_mult + means + r_shift
            port_rets = (stressed_paths @ weights) * leverage - monthly_interest
            
            # Get strategy
            u_rate = upper_var.get() / 100.0
            l_rate = lower_var.get() / 100.0
            
            strategy_map = {
                "Fixed % Initial": fixed_pct_initial(start_capital_var.get(), u_rate, scale=12),
                "Fixed % Current": pct_of_current_capital(u_rate, scale=12),
                "Guardrail 2.5-5%": guardrail_pct(l_rate, u_rate, scale=12),
                "Bucket Strategy": bucket_strategy(cash_years=cash_years_var.get(),
                                                  annual_withdrawal=annual_withdrawal_var.get(),
                                                  scale=12)
            }
            strategy = strategy_map.get(strat_var.get(), strategy_map["Fixed % Initial"])
            
            # Run simulation
            N_MONTHS = simulation_data['N_MONTHS']
            n_bootstrap = n_bootstrap_var.get()
            wealth_paths = np.zeros((n_bootstrap, N_MONTHS + 1))
            wealth_paths[:, 0] = start_capital_var.get()

            '''
            # 
            for b in range(n_bootstrap):
                w = start_capital_var.get()
                for t in range(N_MONTHS):
                    w *= (1 + port_rets[b, t])
                    if t >= 0:
                        w -= strategy(w, year=t/12, inflation_rate=inflation)
                    if w <= 0:
                        w = 0
                        break
                    wealth_paths[b, t+1] = w
            '''

            #'''
            # Vectorised bootstrap
            growth_factors = 1 + port_rets # Pre-calculate (n_bootstrap, N_MONTHS) matrix

            for t in range(N_MONTHS):
                # Update ALL 10,000 paths for month 't' in one line
                wealth_paths[:, t+1] = wealth_paths[:, t] * growth_factors[:, t]
                
                # Calculate withdrawals for ALL paths at once
                # (Requires strategy to handle a NumPy array)
                w_amt = strategy(wealth_paths[:, t+1], year=t/12, inflation_rate=inflation)
                wealth_paths[:, t+1] -= w_amt
                
                # Instant "Floor at Zero" for all paths
                wealth_paths[:, t+1] = np.maximum(wealth_paths[:, t+1], 0)
            #'''

            # Visualization
            xs = np.arange(N_MONTHS + 1) / 12
            
            for i in range(min(100, n_bootstrap)):
                path = wealth_paths[i]
                z_idx = np.where(path <= 0)[0]
                if len(z_idx) > 0:
                    ax.plot(xs[:z_idx[0]+1], path[:z_idx[0]+1], alpha=0.15, lw=0.7)
                    ax.scatter(xs[z_idx[0]], 0, color=colors["fail_marker"], marker='x', s=30)
                else:
                    ax.plot(xs, path, alpha=0.15, lw=0.7)

            p5_path, p95_path = np.percentile(wealth_paths, [5, 95], axis=0)
            ax.plot(xs, np.median(wealth_paths, axis=0), color=colors["line"], 
                   alpha=0.8, lw=2.5, label="Median Path")
            ax.fill_between(xs, p5_path, p95_path, color=colors["envelope"], 
                           alpha=0.15, label="5th - 95th Percentile")
            
            # Stats
            fail_rate = (wealth_paths[:, -1] <= 0).mean() * 100
            p16, p50, p84 = np.percentile(wealth_paths[:, -1], [16, 50, 84])
            prob_target = (wealth_paths[:, -1] >= target_wealth_var.get()).mean() * 100

            summary_txt = (
                f"--- {strat_var.get().upper()} ---\n"
                f"Fail Rate: {fail_rate:>5.1f}%\n"
                f"P(Estate ≥ Target): {prob_target:>4.1f}%\n\n"
                f"--- FINAL WEALTH ---\n"
                f"P16: {portfolio_obj.base_currency}{p16/1e6:>7.2f}M\n"
                f"P50: {portfolio_obj.base_currency}{p50/1e6:>7.2f}M\n"
                f"P84: {portfolio_obj.base_currency}{p84/1e6:>7.2f}M"
            )
            stats_card.config(text=summary_txt)

            ax.yaxis.set_major_formatter(mtick.StrMethodFormatter(f'{portfolio_obj.base_currency} {{x:,.0f}}'))
            padding = max(start_capital_var.get(), np.max(p95_path)) * 0.1
            ax.set_ylim(np.min(p5_path) - padding, np.max(p95_path) + padding)
            ax.set_title(f"Monte Carlo: {strat_var.get()}", loc='center', fontweight='bold')
            ax.set_xlabel("Years in Retirement")
            ax.legend(loc="upper left")
            ax.grid(alpha=0.3)

        fig.tight_layout(pad=2.0)
        canvas.draw()

    # --- EVENT HANDLERS ---
    def apply_settings():
        """Recompute simulation with new settings."""
        recompute_simulation()
        update_plot()

    def schedule_update(delay=350):
        """Debounced update for slider changes."""
        nonlocal update_job
        if update_job is not None:
            root.after_cancel(update_job)
        update_job = root.after(delay, update_plot)

    def on_strat_selected(event):
        """Handle strategy selection."""
        strategy_info = {
            "Fixed % Initial": True,
            "Fixed % Current": False,
            "Guardrail 2.5-5%": False,
            "Bucket Strategy": True,
        }
        
        if strategy_info.get(strat_var.get(), False):
            inflation_slider.config(state="normal")
        else:
            inflation_var.set(0.0)
            inflation_slider.config(state="disabled")
        
        update_plot()

    # --- BINDINGS ---
    strat_menu.bind("<<ComboboxSelected>>", on_strat_selected)
    ret_shift_var.trace_add("write", lambda *a: schedule_update())
    vol_mult_var.trace_add("write", lambda *a: schedule_update())
    inflation_var.trace_add("write", lambda *a: schedule_update())

    # Apply button
    apply_button = ttk.Button(sidebar, text="Apply & Run", command=apply_settings)
    apply_button.pack(fill="x", pady=(10, 5))

    # Initial run
    recompute_simulation()
    update_plot()
    
    return None