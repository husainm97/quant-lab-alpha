import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import sys

# Ensure local imports work
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
from portfolio_module import Portfolio

def markowitz_frontier_random(returns: pd.DataFrame, risk_free: float = 0.02, n_portfolios: int = 10000):
    """
    Generate random long-only portfolios to populate the efficient frontier.
    Returns a dict with returns, vols, Sharpe ratios, and tangent portfolio info.
    """
    rf_annual = 0.02
    risk_free = (1 + rf_annual)**(1/12) - 1  # ~0.00165 (~0.165% per month)
    mean_rets = returns.mean().values
    cov = returns.cov().values
    n = len(mean_rets)

    # Generate random weights
    weights = np.random.rand(n_portfolios, n)
    weights /= weights.sum(axis=1)[:, None]  # normalize to sum to 1

    port_rets = weights @ mean_rets
    port_vols = np.sqrt(np.einsum('ij,jk,ik->i', weights, cov, weights))
    sharpe = (port_rets - risk_free) / port_vols

    idx_tangent = np.argmax(sharpe)
    tangent_w = weights[idx_tangent]
    tangent_ret = port_rets[idx_tangent]
    tangent_vol = port_vols[idx_tangent]

    return {
        "port_rets": port_rets,
        "port_vols": port_vols,
        "sharpe": sharpe,
        "weights": weights,
        "tangent_w": tangent_w,
        "tangent_ret": tangent_ret,
        "tangent_vol": tangent_vol
    }


def plot_markowitz_gui(root, portfolio_obj: Portfolio, risk_free: float = 0.02):

    returns = portfolio_obj.get_common_monthly_returns()
    data = markowitz_frontier_random(returns, risk_free)

    rf_annual = 0.02
    risk_free = (1 + rf_annual)**(1/12) - 1  # ~0.00165 (~0.165% per month)
    fig, ax = plt.subplots(figsize=(8,6))

    # Scatter all random portfolios
    ax.scatter(data["port_vols"], data["port_rets"], c=(data["sharpe"]), cmap='viridis', s=10, alpha=0.5)
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("Sharpe Ratio")

    # Tangent portfolio
    ax.scatter(data["tangent_vol"], data["tangent_ret"], color='r', s=100, label="Tangent Portfolio")

    # Draw tangent line from risk-free rate
    rf = risk_free
    slope = (data["tangent_ret"] - rf) / data["tangent_vol"]
    x_line = np.linspace(0, data["tangent_vol"]*1.5, 50)
    y_line = rf + slope * x_line
    ax.plot(x_line, y_line, 'r--', label="Capital Market Line")

    # Annotate tangent portfolio weights
    weights_text = "\n".join(f"{col}: {w:.2f}" for col, w in zip(returns.columns, data["tangent_w"]))
    ax.annotate(weights_text,
                 xy=(data["tangent_vol"], data["tangent_ret"]),
                 xytext=(data["tangent_vol"]*1.1, data["tangent_ret"]*0.95),
                 arrowprops=dict(facecolor='black', arrowstyle="->"),
                 fontsize=10, bbox=dict(boxstyle='round,pad=0.3', alpha=0.2))

    ax.set_xlabel("Volatility (σ)")
    ax.set_ylabel("Expected Return")
    ax.set_title("Markowitz Efficient Frontier & Tangent Portfolio")
    ax.legend()
    ax.grid(True)

    # Embed in Tkinter
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

    # Buttons frame
    btn_frame = ttk.Frame(root)
    btn_frame.pack(fill="x", pady=5)

    def apply_weights():
        # Apply tangent weights to portfolio
        for i, col in enumerate(returns.columns):
            portfolio_obj.constituents[col] = data["tangent_w"][i]
        # Normalize just in case
        total = sum(portfolio_obj.constituents.values())
        portfolio_obj.constituents = {k: v/total for k,v in portfolio_obj.constituents.items()}
        root.destroy()
        print("Applied tangent weights:", portfolio_obj.constituents)

    ttk.Button(btn_frame, text="Apply Tangent Weights", command=apply_weights).pack(side="left", padx=10)
    ttk.Button(btn_frame, text="Close", command=root.destroy).pack(side="right", padx=10)