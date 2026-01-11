import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import sys

# Maintaining your constraint for older/stable dependencies 
# Scikit-learn's LedoitWolf is the industry standard for shrinkage
from sklearn.covariance import LedoitWolf

# Ensure local imports work
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
from portfolio_module import Portfolio
from gui import PortfolioGUI

def calculate_shrunk_covariance(returns: pd.DataFrame):
    """
    Applies Ledoit-Wolf shrinkage to the covariance matrix.
    Reduces the impact of estimation error in the sample covariance.
    """
    lw = LedoitWolf()
    shrunk_cov = lw.fit(returns).covariance_
    return shrunk_cov

def markowitz_frontier_robust(returns: pd.DataFrame, risk_free_annual: float = 0.02, n_portfolios: int = 10000, n_simulations: int = 50):
    """
    Professional grade MVO using Ledoit-Wolf shrinkage and Resampled Efficiency.
    """
    rf_monthly = (1 + risk_free_annual)**(1/12) - 1
    n_assets = returns.shape[1]
    
    # 1. Ledoit-Wolf Shrinkage
    shrunk_cov = calculate_shrunk_covariance(returns)
    mean_rets = returns.mean().values

    # 2. Resampled Efficiency (Addressing Model Instability)
    # We perturb the returns based on the distribution to find stable weights
    resampled_weights = []
    
    for _ in range(n_simulations):
        # Simulate new mean returns based on the distribution (Perturbation)
        perturbed_means = np.random.multivariate_normal(mean_rets, shrunk_cov / len(returns))
        
        # Generate random portfolios to find the tangent for THIS simulation
        w = np.random.rand(n_portfolios, n_assets)
        w /= w.sum(axis=1)[:, None]
        
        p_rets = w @ perturbed_means
        p_vols = np.sqrt(np.einsum('ij,jk,ik->i', w, shrunk_cov, w))
        p_sharpe = (p_rets - rf_monthly) / p_vols
        
        resampled_weights.append(w[np.argmax(p_sharpe)])

    # Average the weights across simulations (Resampled Tangent Portfolio)
    stable_tangent_w = np.mean(resampled_weights, axis=0)
    
    # Calculate performance for the scatter plot (using base shrunk metrics)
    weights = np.random.rand(n_portfolios, n_assets)
    weights /= weights.sum(axis=1)[:, None]
    
    port_rets = weights @ mean_rets
    port_vols = np.sqrt(np.einsum('ij,jk,ik->i', weights, shrunk_cov, weights))
    sharpe = (port_rets - rf_monthly) / port_vols

    # Calculate metrics for the Stable Tangent Portfolio
    tangent_ret = stable_tangent_w @ mean_rets
    tangent_vol = np.sqrt(stable_tangent_w @ shrunk_cov @ stable_tangent_w)

    return {
        "port_rets": port_rets,
        "port_vols": port_vols,
        "sharpe": sharpe,
        "weights": weights,
        "tangent_w": stable_tangent_w,
        "tangent_ret": tangent_ret,
        "tangent_vol": tangent_vol,
        "rf_monthly": rf_monthly
    }


def plot_markowitz_gui(root, portfolio_obj: Portfolio, risk_free: float = 0.02, gui: PortfolioGUI = None, is_dark = False):

    # 1. Define Colors based on Theme
    if is_dark:
        plt.style.use('dark_background')
        bg_color = "#1c1c1c"  # Match sv_ttk
        text_color = "white"
        box_bg = "#333333"
        cml_color = "white"
        grid_alpha = 0.2
    else:
        plt.style.use('default')
        bg_color = "#fafafa" 
        text_color = "black"
        box_bg = "wheat"
        cml_color = "black"
        grid_alpha = 0.6

    # 2. Get Data
    returns = portfolio_obj.get_common_monthly_returns()
    data = markowitz_frontier_robust(returns, risk_free)
    rf = data["rf_monthly"]

    # 3. Create Single Figure/Axis
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    # 4. Background Cloud
    scatter = ax.scatter(data["port_vols"], data["port_rets"], 
                         c=data["sharpe"], cmap='plasma', s=8, alpha=0.3)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Ex-Ante Sharpe Ratio", color=text_color)
    cbar.ax.yaxis.set_tick_params(color=text_color)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=text_color)

    # 5. Tangent & CML
    ax.scatter(data["tangent_vol"], data["tangent_ret"], 
               color='red', edgecolors='white', s=150, zorder=5, 
               label="Stable Tangent (Resampled)")

    slope = (data["tangent_ret"] - rf) / data["tangent_vol"]
    x_line = np.linspace(0, data["port_vols"].max(), 100)
    y_line = rf + slope * x_line
    ax.plot(x_line, y_line, color=cml_color, linestyle='--', alpha=0.7, label="CML")

    # 6. Annotation and Styling
    weights_text = "Weights:\n" + "\n".join(f"{col}: {w:.1%}" for col, w in zip(returns.columns, data["tangent_w"]))
    ax.text(0.05, 0.5, weights_text, transform=ax.transAxes, verticalalignment='center',
            color=text_color,
            bbox=dict(boxstyle='round', facecolor=box_bg, alpha=0.7, edgecolor=text_color))
    
    ax.set_xlabel("Monthly Volatility (Shrunk)", color=text_color)
    ax.set_ylabel("Expected Monthly Return", color=text_color)
    ax.set_title("Mean-Variance Optimization", color=text_color, fontsize=12)
    ax.tick_params(colors=text_color)

    # 7. Embedding
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

    # 8. Grab current portfolio risk-return
    def plot_portfolio_marker():
        # Remove previous blue dot(s)
        for coll in ax.collections[:]:
            if getattr(coll, "_label", "") == "Your Portfolio":
                coll.remove()
        
        # Compute correct metrics
        mean_rets = returns.mean().values
        shrunk_cov = calculate_shrunk_covariance(returns)
        weights = np.array([portfolio_obj.constituents[col] for col in returns.columns])

        port_mean = weights @ mean_rets                 # Expected monthly return
        port_vol = np.sqrt(weights @ shrunk_cov @ weights)  # Portfolio volatility

        # Plot the portfolio
        ax.scatter(port_vol, port_mean, color='blue', edgecolors='white', s=120, zorder=6, label="Your Portfolio")
        ax.legend(loc='upper left', frameon=True, facecolor=bg_color, edgecolor=text_color)
        canvas.draw()

    plot_portfolio_marker()
    ax.grid(True, linestyle=':', alpha=grid_alpha)
    
    plt.tight_layout()

    btn_frame = ttk.Frame(root)
    btn_frame.pack(fill="x", pady=5)

    def apply_weights():
        # Apply tangent weights to portfolio object
        for i, col in enumerate(returns.columns):
            portfolio_obj.constituents[col] = data["tangent_w"][i]
        
        # Normalize
        total = sum(portfolio_obj.constituents.values())
        portfolio_obj.constituents = {k: v / total for k, v in portfolio_obj.constituents.items()}

        # --- UPDATE TREEVIEW ---
        for child in gui.tree.get_children():  # iterate all rows
            ticker = gui.tree.item(child)["values"][1]  # second column is Ticker
            if ticker in portfolio_obj.constituents:
                new_alloc = portfolio_obj.constituents[ticker] * 100
                values = list(gui.tree.item(child)["values"])
                values[2] = f"{new_alloc:.1f}%"
                gui.tree.item(child, values=values)

        gui._update_total_allocation()  # refresh total allocation label
        plot_portfolio_marker()
        #root.destroy()
        


    ttk.Button(btn_frame, text="Apply Weights", command=apply_weights).pack(side="left", padx=10)
    ttk.Button(btn_frame, text="Cancel", command=root.destroy).pack(side="right", padx=10)