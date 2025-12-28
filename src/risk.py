import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import statsmodels.api as sm
import os
import sys

# Maintain project paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.fetch_ff5 import fetch_ff5_monthly

def calculate_risk_metrics(returns: pd.Series, weights: np.array, cov_matrix: np.array, alpha: float = 0.05):
    # Standard metrics
    sorted_returns = np.sort(returns)
    index = int(alpha * len(sorted_returns))
    var_95 = -sorted_returns[index]
    cvar_95 = -sorted_returns[:index].mean()

    cum_rets = (1 + returns).cumprod()
    running_max = cum_rets.cummax()
    drawdowns = (cum_rets - running_max) / running_max
    max_drawdown = drawdowns.min()

    # Asset Contribution to Volatility
    port_vol = np.sqrt(weights.T @ cov_matrix @ weights)
    marginal_contrib = (cov_matrix @ weights) / port_vol
    pct_contrib = (weights * marginal_contrib) / port_vol

    return {
        "VaR_95": var_95, "CVaR_95": cvar_95,
        "Max_Drawdown": max_drawdown, "Volatility": port_vol,
        "Pct_Contrib": pct_contrib, "Drawdown_Series": drawdowns
    }

def calculate_factor_risk(port_returns, ff_df):
    common_idx = port_returns.index.intersection(ff_df.index)
    y = port_returns.loc[common_idx] - ff_df.loc[common_idx, "RF"]
    factors = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    X = ff_df.loc[common_idx, factors]
    
    model = sm.OLS(y, sm.add_constant(X)).fit()
    betas = model.params[factors].values
    f_cov = X.cov().values
    
    total_var = y.var()
    f_contrib = (betas * (f_cov @ betas)) / total_var
    return pd.Series(f_contrib, index=factors), (y.var() - model.resid.var()) / y.var()

def plot_risk_dashboard(root, portfolio_obj):
    # --- Data Prep ---
    returns_df = portfolio_obj.get_common_monthly_returns()
    tickers = list(returns_df.columns)
    weights = np.array([portfolio_obj.constituents[t] for t in tickers])
    port_returns = returns_df @ weights
    cov_matrix = returns_df.cov().values

    # --- Calculations ---
    m = calculate_risk_metrics(port_returns, weights, cov_matrix)
    ff_df = fetch_ff5_monthly()
    ff_df.index = ff_df.index.to_period("M").to_timestamp("M")
    f_contrib, f_ratio = calculate_factor_risk(port_returns, ff_df)
    asset_contrib = pd.Series(m["Pct_Contrib"], index=tickers)

    # --- Main Figure Setup ---
    fig = plt.figure(figsize=(11, 8.5), dpi=100)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])

    # Plot 1: Drawdown (Bright Red)
    ax_draw = fig.add_subplot(gs[0, :])
    ax_draw.fill_between(m["Drawdown_Series"].index, m["Drawdown_Series"], 0, color='#ff0000', alpha=0.4)
    ax_draw.plot(m["Drawdown_Series"].index, m["Drawdown_Series"], color='#ff0000', linewidth=1)
    ax_draw.set_title("Historical Drawdown Profile (Underwater Chart)", fontweight='bold', fontsize=12)
    ax_draw.grid(True, linestyle=':', alpha=0.6)
    
    # Plot 2: Distribution & Tail Risk (Bottom Right)
    ax_dist = fig.add_subplot(gs[1, 1])
    ax_dist.hist(port_returns, bins=35, color='#3498db', alpha=0.5, edgecolor='white')
    ax_dist.axvline(-m["VaR_95"], color='orange', ls='--', lw=2, label=f'VaR: {m["VaR_95"]:.1%}')
    ax_dist.axvline(-m["CVaR_95"], color='red', lw=2, label=f'CVaR: {m["CVaR_95"]:.1%}')
    ax_dist.set_title("Return Distribution & Tail Risk", fontweight='bold')
    ax_dist.legend()

    # Plot 3: Dynamic Bar Chart (Bottom Left)
    ax_dyn = fig.add_subplot(gs[1, 0])

    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib.cm as cm

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill="both", expand=True)

    view_mode = tk.StringVar(value="Asset Level")

    # --- Color definitions ---

    # Consistent FF5 factor colors (must match regression plots)
    factor_colors = {
        "Mkt-RF": "#1f77b4",  # blue
        "SMB":    "#ff7f0e",  # orange
        "HML":    "#2ca02c",  # green
        "RMW":    "#d62728",  # red
        "CMA":    "#9467bd",  # purple
    }

    # Distinct asset colors (no overlap with factors)
    asset_cmap = cm.get_cmap("tab10")
    asset_colors = {
        asset: asset_cmap(i % 10)
        for i, asset in enumerate(asset_contrib.index)
    }


    def refresh_plot():
        ax_dyn.clear()

        if view_mode.get() == "Asset Level":
            data = asset_contrib.sort_values()
            colors = [asset_colors[a] for a in data.index]
            title = "Asset Risk Contribution"

        else:
            data = f_contrib.sort_values()
            colors = [factor_colors[f] for f in data.index]
            title = "FF5 Factor Risk Attribution"

        ax_dyn.barh(
            data.index,
            data.values,
            color=colors,
            edgecolor="black",
            alpha=0.85
        )

        ax_dyn.set_title(title, fontweight="bold")
        ax_dyn.set_xlabel("Contribution to Portfolio Variance")
        ax_dyn.axvline(0, color="black", lw=0.8)
        ax_dyn.grid(axis="x", linestyle="--", alpha=0.4)

        fig.tight_layout()
        canvas.draw()


    # --- Professional Control & Summary Panel ---
    ctrl_panel = ttk.Frame(root)
    ctrl_panel.pack(fill="x", side="bottom", pady=10)

    # Choice Selection (Left Side)
    choice_frame = ttk.Frame(ctrl_panel)
    choice_frame.pack(side="left", padx=20)

    for text in ["Asset Level", "Factor Level"]:
        ttk.Radiobutton(
            choice_frame,
            text=text,
            variable=view_mode,
            value=text,
            command=refresh_plot
        ).pack(side="left", padx=10)

    # Summary Stats (Middle)
    stats_frame = ttk.Frame(ctrl_panel)
    stats_frame.pack(side="left", expand=True)

    summary_data = [
        ("Monthly Vol", f"{m['Volatility']:.2%}"),
        ("Max Drawdown", f"{m['Max_Drawdown']:.2%}"),
        ("95%-VaR", f"{m['VaR_95']:.2%}"),
        ("95%CVaR", f"{m['CVaR_95']:.2%}")
    ]

    for i, (label, val) in enumerate(summary_data):
        ttk.Label(
            stats_frame,
            text=label,
            font=("Helvetica", 10, "bold")
        ).grid(row=0, column=i * 2, padx=5)

        color = "red" if any(k in label for k in ["Drawdown", "VaR", "CVaR"]) else "black"

        ttk.Label(
            stats_frame,
            text=val,
            foreground=color,
            font=("Helvetica", 10)
        ).grid(row=0, column=i * 2 + 1, padx=15)

    # Close button (Right)
    ttk.Button(
        ctrl_panel,
        text="Close Report",
        command=root.destroy
    ).pack(side="right", padx=20)

    # Initial render
    refresh_plot()
