import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import sys
from sklearn.covariance import LedoitWolf

# --- Original Logic Functions ---

def calculate_shrunk_covariance(returns: pd.DataFrame):
    lw = LedoitWolf()
    return lw.fit(returns).covariance_

def markowitz_frontier_robust(returns: pd.DataFrame, risk_free_annual: float = 0.02, n_portfolios: int = 10000, n_simulations: int = 50):
    rf_monthly = (1 + risk_free_annual)**(1/12) - 1
    n_assets = returns.shape[1]
    shrunk_cov = calculate_shrunk_covariance(returns)
    mean_rets = returns.mean().values

    resampled_weights = []
    for _ in range(n_simulations):
        perturbed_means = np.random.multivariate_normal(mean_rets, shrunk_cov / len(returns))
        w = np.random.rand(n_portfolios, n_assets)
        w /= w.sum(axis=1)[:, None]
        p_rets = w @ perturbed_means
        p_vols = np.sqrt(np.einsum('ij,jk,ik->i', w, shrunk_cov, w))
        p_sharpe = (p_rets - rf_monthly) / p_vols
        resampled_weights.append(w[np.argmax(p_sharpe)])

    stable_tangent_w = np.mean(resampled_weights, axis=0)
    
    weights = np.random.rand(n_portfolios, n_assets)
    weights /= weights.sum(axis=1)[:, None]
    
    port_rets = weights @ mean_rets
    port_vols = np.sqrt(np.einsum('ij,jk,ik->i', weights, shrunk_cov, weights))
    sharpe = (port_rets - rf_monthly) / port_vols

    return {
        "port_rets": port_rets, "port_vols": port_vols, "sharpe": sharpe,
        "weights": weights, "tangent_w": stable_tangent_w,
        "tangent_ret": stable_tangent_w @ mean_rets,
        "tangent_vol": np.sqrt(stable_tangent_w @ shrunk_cov @ stable_tangent_w),
        "rf_monthly": rf_monthly,
        "tickers": returns.columns.tolist()
    }

# --- Interactive Class ---

class InteractiveFrontier:
    def __init__(self, root, portfolio_obj, risk_free, gui, is_dark):
        self.root = root
        self.portfolio_obj = portfolio_obj
        self.gui = gui
        self.is_dark = is_dark
        self.returns = portfolio_obj.get_common_monthly_returns()
        self.data = markowitz_frontier_robust(self.returns, risk_free)
        
        # State for Undo
        self.previous_weights = None
        
        # 1. Restore exact color logic
        if is_dark:
            plt.style.use('dark_background')
            self.bg_color = "#1c1c1c"
            self.text_color = "white"
            self.box_bg = "#333333"
            self.cml_color = "white"
            self.grid_alpha = 0.2
        else:
            plt.style.use('default')
            self.bg_color = "#fafafa" 
            self.text_color = "black"
            self.box_bg = "wheat"
            self.cml_color = "black"
            self.grid_alpha = 0.6

        # 2. UI Setup
        self.fig, self.ax = plt.subplots(figsize=(9, 7), dpi=100)
        self.fig.patch.set_facecolor(self.bg_color)
        self.ax.set_facecolor(self.bg_color)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # 3. Interactions: Annotations and Hover
        self.hover_annot = self.ax.annotate("", xy=(0,0), xytext=(15,15),
                                            textcoords="offset points",
                                            color=self.text_color,
                                            fontsize=8,
                                            bbox=dict(boxstyle='round', facecolor=self.box_bg, alpha=0.9),
                                            arrowprops=dict(arrowstyle="->", color=self.text_color))
        self.hover_annot.set_visible(False)

        self.draw_plot()
        self.plot_portfolio_marker()
        
        # Connect Hover Event
        self.canvas.mpl_connect("motion_notify_event", self.on_hover)

        # Buttons
        self.btn_frame = ttk.Frame(root)
        self.btn_frame.pack(fill="x", pady=5)
        
        self.apply_btn = ttk.Button(self.btn_frame, text="Apply Optimized Weights", command=self.apply_weights)
        self.apply_btn.pack(side="left", padx=10)
        
        self.undo_btn = ttk.Button(self.btn_frame, text="Undo Changes", command=self.undo_weights, state=tk.DISABLED)
        self.undo_btn.pack(side="left", padx=10)
        
        ttk.Button(self.btn_frame, text="Close", command=root.destroy).pack(side="right", padx=10)

    def draw_plot(self):
        d = self.data
        
        # Background Cloud
        
        self.scatter = self.ax.scatter(d["port_vols"], d["port_rets"], 
                                       c=d["sharpe"], cmap='plasma', s=8, alpha=0.3)
        
        cbar = self.fig.colorbar(self.scatter, ax=self.ax)
        cbar.set_label("Ex-Ante Sharpe Ratio", color=self.text_color)
        cbar.ax.yaxis.set_tick_params(color=self.text_color)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=self.text_color)

        # Tangent & CML
        self.ax.scatter(d["tangent_vol"], d["tangent_ret"], 
                        color='red', edgecolors='white', s=150, zorder=5, 
                        label="Stable Tangent (Resampled)")

        slope = (d["tangent_ret"] - d["rf_monthly"]) / d["tangent_vol"]
        x_line = np.linspace(0, d["port_vols"].max(), 100)
        y_line = d["rf_monthly"] + slope * x_line
        self.ax.plot(x_line, y_line, color=self.cml_color, linestyle='--', alpha=0.7, label="CML")

        # Fixed Weights Box (Sharpe + Weights only)
        tangent_sharpe = (d["tangent_ret"] - d["rf_monthly"]) / d["tangent_vol"]
        w_text = f"Optimized Sharpe: {tangent_sharpe:.3f}\n" + "-"*20 + "\n"
        w_text += "\n".join(f"{col}: {w:.1%}" for col, w in zip(d["tickers"], d["tangent_w"]))
        
        self.ax.text(0.05, 0.5, w_text, transform=self.ax.transAxes, verticalalignment='center',
                     color=self.text_color, fontsize=9,
                     bbox=dict(boxstyle='round', facecolor=self.box_bg, alpha=0.7, edgecolor=self.text_color))
        
        self.ax.set_xlabel("Monthly Volatility (Risk)", color=self.text_color)
        self.ax.set_ylabel("Expected Monthly Return", color=self.text_color)
        self.ax.tick_params(colors=self.text_color)
        self.ax.grid(True, linestyle=':', alpha=self.grid_alpha)

    def plot_portfolio_marker(self):
        # Force removal of existing marker by checking labels in collections
        for artist in list(self.ax.collections):
            if artist.get_label() == "Current Portfolio":
                artist.remove()
        
        mean_rets = self.returns.mean().values
        shrunk_cov = calculate_shrunk_covariance(self.returns)
        # Ensure weights are aligned with the ticker order used in optimization
        weights = np.array([self.portfolio_obj.constituents[col] for col in self.data["tickers"]])

        port_mean = weights @ mean_rets
        port_vol = np.sqrt(weights @ shrunk_cov @ weights)

        self.ax.scatter(port_vol, port_mean, color='blue', edgecolors='white', 
                        s=150, zorder=10, label="Current Portfolio")
        
        # Refresh Legend
        self.ax.legend(loc='upper left', frameon=True, facecolor=self.bg_color, edgecolor=self.text_color)
        self.canvas.draw()

    def on_hover(self, event):
        if event.inaxes == self.ax:
            cont, ind = self.scatter.contains(event)
            if cont:
                idx = ind["ind"][0]
                pos = self.scatter.get_offsets()[idx]
                sharpe = self.data["sharpe"][idx]
                weights = self.data["weights"][idx]
                
                # Formulate hover text: Sharpe + Weights
                txt = f"Sharpe: {sharpe:.3f}\n" + "-"*15 + "\n"
                txt += "\n".join(f"{tkr}: {w:.1%}" for tkr, w in zip(self.data["tickers"], weights))
                
                self.hover_annot.xy = pos
                self.hover_annot.set_text(txt)
                self.hover_annot.set_visible(True)
                self.canvas.draw_idle()
            else:
                if self.hover_annot.get_visible():
                    self.hover_annot.set_visible(False)
                    self.canvas.draw_idle()

    def _sync_to_gui(self):
        """Helper to push portfolio_obj state to the main GUI Treeview."""
        if self.gui:
            for child in self.gui.tree.get_children():
                ticker = self.gui.tree.item(child)["values"][1]
                if ticker in self.portfolio_obj.constituents:
                    new_alloc = self.portfolio_obj.constituents[ticker] * 100
                    values = list(self.gui.tree.item(child)["values"])
                    values[2] = f"{new_alloc:.1f}%"
                    self.gui.tree.item(child, values=values)
            self.gui._update_total_allocation()

    def apply_weights(self):
        # Store current state for Undo before applying new ones
        self.previous_weights = self.portfolio_obj.constituents.copy()
        
        # Apply Tangent weights
        for i, col in enumerate(self.data["tickers"]):
            self.portfolio_obj.constituents[col] = self.data["tangent_w"][i]
        
        # Re-normalize
        total = sum(self.portfolio_obj.constituents.values())
        self.portfolio_obj.constituents = {k: v / total for k, v in self.portfolio_obj.constituents.items()}

        self._sync_to_gui()
        self.plot_portfolio_marker()
        
        # Toggle buttons
        self.apply_btn.config(state=tk.DISABLED)
        self.undo_btn.config(state=tk.NORMAL)

    def undo_weights(self):
        if self.previous_weights:
            self.portfolio_obj.constituents = self.previous_weights.copy()
            self._sync_to_gui()
            self.plot_portfolio_marker()
            
            # Reset buttons
            self.previous_weights = None
            self.apply_btn.config(state=tk.NORMAL)
            self.undo_btn.config(state=tk.DISABLED)

# --- Entry Point ---
def plot_markowitz_gui(root, portfolio_obj, risk_free=0.02, gui=None, is_dark=False):
    InteractiveFrontier(root, portfolio_obj, risk_free, gui, is_dark)