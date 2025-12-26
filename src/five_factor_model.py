# ff5_analyser.py

import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import zipfile
import io
import requests
import numpy as np
import os, sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from simulations.portfolio_module import Portfolio
from data.fetch_ff5 import fetch_ff5_monthly

# -----------------------------
#  Utility Functions
# -----------------------------

def prepare_etf(df, etf_col):
    """
    Prepare ETF returns DataFrame with columns ['ETF_TV','Market','SMB','HML','RMW','CMA','RF']
    """
    etf = df[[etf_col,'Mkt-RF','SMB','HML','RMW','CMA','RF']].copy()
    etf.rename(columns={etf_col:'ETF_TV','Mkt-RF':'Market'}, inplace=True)
    return etf

def rolling_factor_loadings_5f(df, window):
    X = sm.add_constant(df[['Mkt-RF','SMB','HML','RMW','CMA']])
    betas = []
    idx = []
    for i in range(window, len(df)):
        ywin = df['excess'].iloc[i-window:i]
        Xwin = X.iloc[i-window:i]
        model = sm.OLS(ywin, Xwin).fit()
        betas.append(model.params.drop('const'))
        idx.append(df.index[i])
    return pd.DataFrame(betas, index=idx)

# -----------------------------
# Main popup logic
# -----------------------------


def ff5_regression_core(portfolio_returns: pd.Series, ff_df: pd.DataFrame):
    factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    common_idx = portfolio_returns.index.intersection(ff_df.index)
    
    if len(common_idx) == 0:
        raise ValueError("No overlapping dates between portfolio returns and FF5 factors!")
    
    y = portfolio_returns.loc[common_idx] - ff_df.loc[common_idx, "RF"]
    X = sm.add_constant(ff_df.loc[common_idx, factor_cols])
    model = sm.OLS(y, X).fit()
    
    resid = model.resid
    return {
        "alpha": model.params["const"],
        "betas": model.params[factor_cols].values,
        "resid_std": resid.std(ddof=1),
        "model": model
    }

def run_ff5_analysis(master, portfolio_obj: Portfolio = None):
    # -----------------------------
    # Portfolio return construction
    # -----------------------------
    MONTHS_PER_YEAR = 12

    returns_df = portfolio_obj.get_common_monthly_returns()
    tickers = list(returns_df.columns)
    weights = np.array([portfolio_obj.constituents[t] for t in tickers]) if portfolio_obj else np.ones(len(tickers))/len(tickers)
    leverage = float(portfolio_obj.leverage) if portfolio_obj else 1.0
    annual_interest = float(portfolio_obj.interest_rate) if portfolio_obj else 0.0
    monthly_borrow_rate = annual_interest / MONTHS_PER_YEAR
    borrow_cost_per_equity = max(0.0, leverage - 1.0) * monthly_borrow_rate

    # weighted simple returns
    weighted_returns = returns_df[tickers].dot(weights)

    # leveraged portfolio equity return
    port_returns = weighted_returns * leverage - borrow_cost_per_equity

    # Make it a DataFrame with a stable name
    monthly = port_returns.to_frame(name="Portfolio")
    asset_col = "Portfolio"

    # -----------------------------
    # Fama–French 5 data
    # -----------------------------
    ff = fetch_ff5_monthly()
    ff.index = ff.index.to_period("M").to_timestamp("M")
    combined = pd.concat([monthly, ff], axis=1, join="inner")

    # -----------------------------
    # Tk window setup
    # -----------------------------
    win = tk.Toplevel(master)
    win.title("Fama–French 5-Factor Analysis")

    # Use a frame to hold plot + text side by side
    main_frame = ttk.Frame(win)
    main_frame.pack(fill=tk.BOTH, expand=True)

    # Figure / plot
    fig = plt.Figure(figsize=(10, 6), dpi=100)
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=main_frame)
    canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Text box on the right
    text_box = tk.Text(main_frame, width=90, height=35)
    text_box.pack(side=tk.RIGHT, fill=tk.Y)

    # -----------------------------
    # UI state variables
    # -----------------------------
    show_3y = tk.BooleanVar(value=False)
    show_5y = tk.BooleanVar(value=True)
    show_10y = tk.BooleanVar(value=False)
    mode = tk.StringVar(value="static")

    factor_colors = {
        "Mkt-RF": "tab:blue",
        "SMB": "tab:orange",
        "HML": "tab:green",
        "RMW": "tab:red",
        "CMA": "tab:purple",
    }

    # -----------------------------
    # Static regression
    # -----------------------------
    def plot_static():
        ax.clear()
        y = combined[asset_col] - combined["RF"]
        X = sm.add_constant(combined[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]])
        model = sm.OLS(y, X).fit()

        betas = model.params.drop("const")
        tstats = model.tvalues.drop("const")

        bar_colors = [factor_colors[f] for f in betas.index]

        ax.bar(betas.index, betas.values, color=bar_colors, edgecolor="black")
        ax.set_title(f"{asset_col} – FF5 Factor Loadings")
        ax.set_ylabel("Beta")
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.grid(alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

        # Update text box
        text_box.delete("1.0", tk.END)
        text_box.insert(tk.END, f"===== {asset_col} Regression =====\n")
        text_box.insert(
            tk.END,
            f"Alpha: {model.params['const']:.4f} (t={model.tvalues['const']:.2f})\n\n"
        )
        for fac in betas.index:
            b = betas[fac]
            t = tstats[fac]
            sig = "SIGNIFICANT" if abs(t) > 2 else "not significant"
            text_box.insert(tk.END, f"{fac}: {b:.4f}  (t={t:.2f}) → {sig}\n")

        text_box.insert(tk.END, model.summary().as_text())
        canvas.draw()

    # -----------------------------
    # Rolling regression
    # -----------------------------
    def plot_rolling():
        ax.clear()
        df = combined.copy()
        df["excess"] = df[asset_col] - df["RF"]
        factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
        Xfull = sm.add_constant(df[factor_cols])

        windows = [
            (36, show_3y.get(), '3y'),
            (60, show_5y.get(), '5y'),
            (120, show_10y.get(), '10y'),
        ]

        plotted_factors = set()
        for W, enabled, label in windows:
            if not enabled:
                continue
            betas = []
            idx = []
            for i in range(W, len(df)):
                y = df["excess"].iloc[i - W:i]
                X = Xfull.iloc[i - W:i]
                m = sm.OLS(y, X).fit()
                betas.append(m.params.drop("const"))
                idx.append(df.index[i])
            roll = pd.DataFrame(betas, index=idx, columns=factor_cols)

            linestyle = '-' if W == 60 else '--'
            alpha_val = 0.9 if W == 60 else 0.5

            for fac in factor_cols:
                ax.plot(
                    roll.index, roll[fac],
                    color=factor_colors[fac],
                    linewidth=1.2,
                    linestyle=linestyle,
                    alpha=alpha_val,
                    label=fac if fac not in plotted_factors else None
                )
                plotted_factors.add(fac)

        ax.set_title(f"{asset_col} – Rolling Factor Loadings")
        ax.grid(alpha=0.3)
        ax.legend(title="Factors")
        canvas.draw()

        text_box.delete("1.0", tk.END)
        text_box.insert(tk.END, f"Rolling regression: {asset_col}\n")
        text_box.insert(tk.END, "Toggle windows to update.\n")

    # -----------------------------
    # update logic
    # -----------------------------
    def update_plot(*_):
        if mode.get() == "static":
            plot_static()
        else:
            plot_rolling()

    # -----------------------------
    # Choices frame below plot (horizontal)
    # -----------------------------
    control_frame = ttk.Frame(win)
    control_frame.pack(fill=tk.X, side=tk.TOP, pady=5)

    ttk.Radiobutton(control_frame, text="Static", variable=mode, value="static", command=update_plot).pack(side=tk.LEFT, padx=5)
    ttk.Radiobutton(control_frame, text="Rolling", variable=mode, value="rolling", command=update_plot).pack(side=tk.LEFT, padx=5)

    ttk.Checkbutton(control_frame, text="3-Year", variable=show_3y, command=update_plot).pack(side=tk.LEFT, padx=5)
    ttk.Checkbutton(control_frame, text="5-Year", variable=show_5y, command=update_plot).pack(side=tk.LEFT, padx=5)
    ttk.Checkbutton(control_frame, text="10-Year", variable=show_10y, command=update_plot).pack(side=tk.LEFT, padx=5)

    # default view
    plot_static()




