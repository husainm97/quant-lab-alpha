# ff5_analyser.py

import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import numpy as np
import os, sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from simulations.portfolio_module import Portfolio
from data.fetch_ff5 import fetch_ff5_monthly

# ----------------------------------------------------------
# Currently used in notebooks
# ----------------------------------------------------------
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
# ----------------------------------------------------------

# -----------------------------
# Main popup logic
# -----------------------------

def run_ff5_analysis(master, portfolio_obj: Portfolio = None, is_dark=False):
    # -----------------------------
    # Theme setup
    # -----------------------------
    if is_dark:
        plt.style.use('dark_background')
        marker_color = 'lightgrey'
    else:
        plt.style.use('default')
        marker_color = 'grey'

    MONTHS_PER_YEAR = 12

    # -----------------------------
    # Portfolio return construction
    # -----------------------------
    all_assets = {}
    tickers = list(portfolio_obj.constituents.keys())
    
    for ticker in tickers:
        ent = portfolio_obj.data.get(ticker)
        if ent and 'prices_usd' in ent:
            price_series = ent['prices_usd']
            asset_returns = portfolio_obj._price_to_monthly_returns(price_series)
            all_assets[ticker] = asset_returns
    
    # returns_df is already a pd.DataFrame per your provided method
    returns_df = portfolio_obj.get_common_monthly_returns(use_usd=True)
    
    # weights aligned to the DataFrame columns
    weights = np.array([portfolio_obj.constituents[t] for t in returns_df.columns])
    leverage = float(portfolio_obj.leverage)
    annual_interest = float(portfolio_obj.interest_rate)
    monthly_borrow_rate = annual_interest / MONTHS_PER_YEAR
    borrow_cost_per_equity = max(0.0, leverage - 1.0) * monthly_borrow_rate

    # Vectorized dot product on the DataFrame
    weighted_returns = returns_df.dot(weights)
    port_returns = weighted_returns * leverage - borrow_cost_per_equity
    all_assets["Portfolio"] = port_returns

    # -----------------------------
    # Fama-French 5 data
    # -----------------------------
    ff = fetch_ff5_monthly(region=portfolio_obj.factor_region)
    ff.index = ff.index.to_period("M").to_timestamp("M")
    
    combined_data = {}
    for asset_name, asset_series in all_assets.items():
        # TOUCHING NOTHING BUT THE FAILSAFE: 
        # Only call .to_frame() if the object is actually a Series.
        # 'Portfolio' and individual assets are Series here.
        if isinstance(asset_series, pd.Series):
            asset_input = asset_series.to_frame(name=asset_name)
        else:
            asset_input = asset_series

        combined_data[asset_name] = pd.concat([asset_input, ff], 
                                              axis=1, join="inner")

    # -----------------------------
    # Tk window setup
    # -----------------------------
    win = tk.Toplevel(master)
    win.title("Fama-French 5-Factor Analysis")

    main_frame = ttk.Frame(win)
    main_frame.pack(fill=tk.BOTH, expand=True)

    fig = plt.Figure(figsize=(10, 6), dpi=100)
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=main_frame)
    canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    stats_frame = ttk.LabelFrame(main_frame, text="Regression Statistics", padding=10)
    
    text_box = tk.Text(stats_frame, width=85, height=35, wrap="word", 
                       font=("Consolas", 9), relief="flat", 
                       borderwidth=0, highlightthickness=0)
    text_box.pack(fill=tk.BOTH, expand=True)
    
    scrollbar = ttk.Scrollbar(stats_frame, command=text_box.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    text_box.config(yscrollcommand=scrollbar.set)

    # Toggle button and state
    stats_visible = tk.BooleanVar(value=False)
    
    def toggle_stats():
        if stats_visible.get():
            stats_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
            toggle_btn.config(text="Hide Details <<")
        else:
            stats_frame.pack_forget()
            toggle_btn.config(text="Show Details >>")
        stats_visible.set(not stats_visible.get())
    


    # -----------------------------
    # UI state variables
    # -----------------------------
    selected_asset = tk.StringVar(value="Portfolio")
    custom_window = tk.DoubleVar(value=5.0)
    show_3y = tk.BooleanVar(value=False)
    show_5y = tk.BooleanVar(value=False)
    show_10y = tk.BooleanVar(value=False)
    mode = tk.StringVar(value="static")

    factor_colors = {
        "Mkt-RF": "tab:blue", "SMB": "tab:orange", "HML": "tab:green",
        "RMW": "tab:red", "CMA": "tab:purple",
    }

    # -----------------------------
    # Static regression
    # -----------------------------
    def plot_static():
        ax.clear()
        asset_col = selected_asset.get()
        combined = combined_data[asset_col]
        
        y = combined[asset_col] - combined["RF"]
        X = sm.add_constant(combined[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]])
        
        
        
        model = sm.OLS(y, X).fit()

        betas = model.params.drop("const")
        tstats = model.tvalues.drop("const")

        ax.bar(betas.index, betas.values, color=[factor_colors[f] for f in betas.index], edgecolor="black")
        ax.set_title(f"{asset_col} - FF5 Factor Loadings")
        ax.set_ylabel("Beta")
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.grid(alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

        text_box.delete("1.0", tk.END)
        text_box.insert(tk.END, f"===== {asset_col} Regression =====\n")
        text_box.insert(tk.END, f"Alpha: {model.params['const']:.4f} (t={model.tvalues['const']:.2f})\n\n")
        for fac in betas.index:
            sig = "SIGNIFICANT" if abs(tstats[fac]) > 2 else "not significant"
            text_box.insert(tk.END, f"{fac}: {betas[fac]:.4f}  (t={tstats[fac]:.2f}) -> {sig}\n")

        text_box.insert(tk.END, "\n" + model.summary().as_text())
        canvas.draw()

    # -----------------------------
    # Rolling regression
    # -----------------------------
    def plot_rolling():
        ax.clear()
        asset_col = selected_asset.get()
        combined = combined_data[asset_col]
        df = combined.copy()

        df["excess"] = df[asset_col] - df["RF"]
        factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
        Xfull = sm.add_constant(df[factor_cols])

        custom_months = int(custom_window.get() * 12)
        windows_to_plot = [(custom_months, "custom")]
        if show_3y.get(): windows_to_plot.append((36, "fixed"))
        if show_5y.get(): windows_to_plot.append((60, "fixed"))
        if show_10y.get(): windows_to_plot.append((120, "fixed"))

        if len(df) < min([w[0] for w in windows_to_plot]):
            ax.text(0.5, 0.5, f"Not Enough Data\n({len(df)} months)", ha='center', va='center', transform=ax.transAxes)
            canvas.draw()
            return

        ax.set_ylabel("Betas")
        ax.set_xlabel("Year")
        plotted_factors = set()
        
        for W, window_type in windows_to_plot:
            if len(df) <= W: continue
            betas = []
            idx = []
            for i in range(W, len(df)):
                m = sm.OLS(df["excess"].iloc[i-W:i], Xfull.iloc[i-W:i]).fit()
                betas.append(m.params.drop("const"))
                idx.append(df.index[i])
            
            roll = pd.DataFrame(betas, index=idx, columns=factor_cols)
            alpha_val = 0.45 + (min(W, 120) / 120) * 0.5
            linewidth = 1.0 + (0.8 * min(W - 12, 108) / 108)
            linestyle = '-' if window_type == "fixed" else (0, (5, 5))

            for fac in factor_cols:
                if window_type == "fixed":
                    ax.plot([roll.index[0], roll.index[-1]], [roll[fac].iloc[0], roll[fac].iloc[-1]],
                            marker='o', markersize=3, color=marker_color, ls='', zorder=10)
                ax.plot(roll.index, roll[fac], color=factor_colors[fac],
                        linewidth=linewidth, linestyle=linestyle, alpha=alpha_val,
                        label=fac if fac not in plotted_factors else None)
                plotted_factors.add(fac)

        ax.set_title(f"{asset_col} - Rolling Factor Loadings")
        ax.grid(alpha=0.3)
        if plotted_factors:
            ax.legend(title="Factors", loc='best', fontsize='small')
        canvas.draw()

        text_box.delete("1.0", tk.END)
        text_box.insert(tk.END, f"Rolling regression: {asset_col}\n\nActive windows:\n")
        for W, wtype in windows_to_plot:
            text_box.insert(tk.END, f"  • {W/12:.1f} years ({W} months) - {wtype}\n")
        
        text_box.insert(tk.END, 
            "\n\nA rolling regression computes coefficients over a moving window of data.\n\n"
            "Visual encoding:\n"
            "• Line opacity: Higher opacity = longer window\n"
            "• Line style: Solid = fixed checkboxes, dashed = custom slider\n"
            "• Grey dots: Endpoints of fixed window regressions\n\n"
            "Interpretation:\n"
            "- Shorter windows (1-3y) react quickly but are noisier\n"
            "- Longer windows (7-10y) are smoother but lag changes\n"
        )

    def update_plot(*_):
        slider_label.config(text=f"{custom_window.get():.1f}y")
        plot_static() if mode.get() == "static" else plot_rolling()

    # -----------------------------
    # Control frame
    # -----------------------------
    control_frame = ttk.Frame(win)
    control_frame.pack(fill=tk.X, side=tk.TOP, pady=5)

    ttk.Label(control_frame, text="Asset:").pack(side=tk.LEFT, padx=5)
    asset_menu = ttk.Combobox(control_frame, textvariable=selected_asset, 
                             values=["Portfolio"] + returns_df.columns.tolist(), state="readonly", width=15)
    asset_menu.pack(side=tk.LEFT, padx=5)
    asset_menu.bind("<<ComboboxSelected>>", update_plot)

    ttk.Radiobutton(control_frame, text="Static", variable=mode, value="static", command=update_plot).pack(side=tk.LEFT, padx=5)
    ttk.Radiobutton(control_frame, text="Rolling", variable=mode, value="rolling", command=update_plot).pack(side=tk.LEFT, padx=5)

    ttk.Separator(control_frame, orient="vertical").pack(side=tk.LEFT, fill='y', padx=10)
    ttk.Scale(control_frame, from_=1, to=10, variable=custom_window, command=update_plot, length=150).pack(side=tk.LEFT, padx=5)
    slider_label = ttk.Label(control_frame, text="5.0y", width=5)
    slider_label.pack(side=tk.LEFT, padx=2)

    ttk.Separator(control_frame, orient="vertical").pack(side=tk.LEFT, fill='y', padx=10)
    for txt, var in [("3y", show_3y), ("5y", show_5y), ("10y", show_10y)]:
        ttk.Checkbutton(control_frame, text=txt, variable=var, command=update_plot).pack(side=tk.LEFT, padx=3)

    ttk.Separator(control_frame, orient="vertical").pack(side=tk.LEFT, fill='y', padx=10)
    toggle_btn = ttk.Button(control_frame, text="Show Details >>", command=toggle_stats)
    toggle_btn.pack(side=tk.LEFT, padx=5)

    selected_asset.trace_add("write", update_plot)
    plot_static()