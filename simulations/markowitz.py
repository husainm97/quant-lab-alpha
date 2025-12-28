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

def plot_markowitz_gui(root, portfolio_obj: Portfolio, risk_free: float = 0.02, gui: PortfolioGUI = None,):
    returns = portfolio_obj.get_common_monthly_returns()
    
    # Use the robust solver
    data = markowitz_frontier_robust(returns, risk_free)
    rf = data["rf_monthly"]

    fig, ax = plt.subplots(figsize=(8,6), dpi=100)
    
    # Background: Random portfolios (visualizing the cloud)
    scatter = ax.scatter(data["port_vols"], data["port_rets"], 
                         c=data["sharpe"], cmap='plasma', s=8, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Ex-Ante Sharpe Ratio")

    # The Robust Tangent Portfolio (Stable result)
    ax.scatter(data["tangent_vol"], data["tangent_ret"], 
               color='red', edgecolors='white', s=150, zorder=5, 
               label="Stable Tangent (Resampled)")

    # Capital Market Line
    slope = (data["tangent_ret"] - rf) / data["tangent_vol"]
    x_line = np.linspace(0, data["port_vols"].max(), 100)
    y_line = rf + slope * x_line
    ax.plot(x_line, y_line, color='black', linestyle='--', alpha=0.7, label="CML")

    # Improved Annotation
    weights_text = "Weights:\n" + "\n".join(f"{col}: {w:.1%}" for col, w in zip(returns.columns, data["tangent_w"]))
    ax.text(0.05, 0.5, weights_text, transform=ax.transAxes, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel("Monthly Volatility (Shrunk)")
    ax.set_ylabel("Expected Monthly Return")
    ax.set_title("Robust Mean-Variance Optimization\n(Ledoit-Wolf + Resampled Efficiency)", fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    # Tkinter Embedding
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

    btn_frame = ttk.Frame(root)
    btn_frame.pack(fill="x", pady=5)

    def apply_weights():
        # Apply tangent weights to portfolio object
        for i, col in enumerate(returns.columns):
            portfolio_obj.constituents[col] = data["tangent_w"][i]
        
        # Normalize
        total = sum(portfolio_obj.constituents.values())
        portfolio_obj.constituents = {k: v / total for k, v in portfolio_obj.constituents.items()}
        print(f"Applied Markowitz weights: {portfolio_obj.constituents}")

        # --- UPDATE TREEVIEW ---
        for child in gui.tree.get_children():  # iterate all rows
            ticker = gui.tree.item(child)["values"][1]  # second column is Ticker
            if ticker in portfolio_obj.constituents:
                new_alloc = portfolio_obj.constituents[ticker] * 100
                values = list(gui.tree.item(child)["values"])
                values[2] = f"{new_alloc:.1f}%"
                gui.tree.item(child, values=values)

        gui._update_total_allocation()  # refresh total allocation label
        root.destroy()


    ttk.Button(btn_frame, text="Apply Robust Weights", command=apply_weights).pack(side="left", padx=10)
    ttk.Button(btn_frame, text="Cancel", command=root.destroy).pack(side="right", padx=10)