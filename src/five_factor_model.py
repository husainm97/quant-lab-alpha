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

# -----------------------------
#  Utility Functions
# -----------------------------
def fetch_ff5_monthly():
    """
    Fetches Fama-French 5-Factor monthly data (2x3 CSV format),
    parses only the monthly rows, converts percentages to decimals,
    and returns a clean DataFrame with datetime index.
    """
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
    r = requests.get(url)
    
    # Open the ZIP file from bytes
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        # Find the CSV file
        csv_filename = next(name for name in z.namelist() if name.lower().endswith('.csv'))
        with z.open(csv_filename) as f:
            lines = f.read().decode('utf-8').splitlines()
    
    # Locate the header row
    header_idx = next(i for i, line in enumerate(lines) if "Mkt-RF" in line and "RF" in line)
    header = lines[header_idx].strip().split(',')
    
    # Collect only the monthly data lines (lines starting with YYYYMM)
    data_lines = []
    for line in lines[header_idx + 1:]:
        if not line.strip() or not line[:6].isdigit():
            break
        data_lines.append(line.strip())
    
    # Create DataFrame
    df = pd.read_csv(io.StringIO("\n".join(data_lines)), names=header)
    
    # Parse date and set as index
    df[header[0]] = pd.to_datetime(df[header[0]], format="%Y%m")
    df.set_index(header[0], inplace=True)
    
    # Convert percentages to decimals
    df = df / 100
    
    return df

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

def run_ff5_analysis(master, portfolio_obj: Portfolio = None):

    # -----------------------------
    # Portfolio return construction
    # -----------------------------
    MONTHS_PER_YEAR = 12
    returns_df = portfolio_obj.get_common_monthly_returns()
    tickers = list(returns_df.columns)
    weights = np.array([portfolio_obj.constituents[t] for t in tickers]) \
            if portfolio_obj else np.ones(N_ASSETS)/N_ASSETS
    leverage = float(portfolio_obj.leverage) if portfolio_obj else 1.0
    annual_interest = float(portfolio_obj.interest_rate) if portfolio_obj else 0.0

    monthly_borrow_rate = annual_interest / MONTHS_PER_YEAR
    borrow_cost_per_equity = max(0.0, leverage - 1.0) * monthly_borrow_rate


    # weighted simple returns
    weighted_returns = returns_df[tickers].dot(weights)

    # leveraged portfolio equity return
    port_returns = weighted_returns * leverage - borrow_cost_per_equity

    # Make it a DataFrame with a stable name
    monthly = port_returns.copy()
    monthly.name = "Portfolio"
    monthly = monthly.to_frame()
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

    fig = plt.Figure(figsize=(10, 6), dpi=100)
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    text_box = tk.Text(win, width=50, height=35)
    text_box.pack(side=tk.RIGHT, fill=tk.Y)

    show_3y = tk.BooleanVar(value=True)
    show_5y = tk.BooleanVar(value=True)
    show_10y = tk.BooleanVar(value=True)
    mode = tk.StringVar(value="static")

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

        factor_colors = {
            "Mkt-RF": "tab:blue",
            "SMB": "tab:orange",
            "HML": "tab:green",
            "RMW": "tab:red",
            "CMA": "tab:purple",
        }
        bar_colors = [factor_colors[f] for f in betas.index]

        ax.bar(betas.index, betas.values, color=bar_colors, edgecolor="black")
        ax.set_title(f"{asset_col} – FF5 Factor Loadings")
        ax.set_ylabel("Beta")
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.grid(alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

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
            (36, show_3y.get()),
            (60, show_5y.get()),
            (120, show_10y.get()),
        ]

        factor_colors = {
            "Mkt-RF": "tab:blue",
            "SMB": "tab:orange",
            "HML": "tab:green",
            "RMW": "tab:red",
            "CMA": "tab:purple",
        }

        for W, enabled in windows:
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

            roll = pd.DataFrame(betas, index=idx)

            for fac in factor_cols:
                ax.plot(
                    roll.index,
                    roll[fac],
                    color=factor_colors[fac],
                    linewidth=1.2,
                    alpha=0.8,
                    label=f"{fac} ({W}m)",
                )

        ax.set_title(f"{asset_col} – Rolling Factor Loadings")
        ax.grid(alpha=0.3)
        canvas.draw()

        text_box.delete("1.0", tk.END)
        text_box.insert(tk.END, f"Rolling regression: {asset_col}\n")
        text_box.insert(tk.END, "Toggle windows to update.\n")

    # -----------------------------
    # update logic + UI
    # -----------------------------
    def update_plot(*_):
        if mode.get() == "static":
            plot_static()
        else:
            plot_rolling()

    frame = ttk.Frame(win)
    frame.pack(fill=tk.X)

    ttk.Radiobutton(frame,
        text="Static Factor Loadings",
        variable=mode, value="static",
        command=update_plot).pack(anchor="w")

    ttk.Radiobutton(frame,
        text="Rolling Betas",
        variable=mode, value="rolling",
        command=update_plot).pack(anchor="w")

    ttk.Checkbutton(frame, text="3-Year Window (36m)",
        variable=show_3y, command=update_plot).pack(anchor="w")

    ttk.Checkbutton(frame, text="5-Year Window (60m)",
        variable=show_5y, command=update_plot).pack(anchor="w")

    ttk.Checkbutton(frame, text="10-Year Window (120m)",
        variable=show_10y, command=update_plot).pack(anchor="w")

    # default view
    plot_static()


