#!/usr/bin/env python3
# gui.py — Quant Lab Alpha basic GUI for factor analysis

"""
This file currently contains a basic tkinter GUI that allows building a portfolio from various sources and choosing leverage and withdrawal strategies. 
Options for Markowitz optimisation, 5 factor analysis and retirement spending simulations are available. We are still to plug the right functions to the buttons!
"""

import tkinter as tk
from tkinter import ttk, messagebox

class PortfolioGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Portfolio Builder")
        self.root.geometry("1200x800")

        self.portfolio = []
        self.strategy_params = {}
        self.leverage = 1.0
        self.interest_rate = 0.0

        self._build_ui()

    # =============================
    # Main UI Builder
    # =============================
    def _build_ui(self):
        self._build_portfolio_section()
        self._build_leverage_section()
        self._build_strategy_section()
        self._build_action_section()

    # =============================
    # Portfolio Section
    # =============================
    def _build_portfolio_section(self):
        frame = ttk.LabelFrame(self.root, text="Build Your Portfolio", padding=10)
        frame.pack(fill="x", padx=10, pady=10)

        ttk.Label(frame, text="Source:").grid(row=0, column=0, padx=5, pady=5)
        self.source_var = tk.StringVar(value="Yahoo")
        ttk.Combobox(frame, textvariable=self.source_var, values=["Yahoo", "Source", "Manual"], width=10).grid(
            row=0, column=1, padx=5, pady=5
        )

        ttk.Label(frame, text="Ticker:").grid(row=0, column=2, padx=5, pady=5)
        self.ticker_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.ticker_var, width=15).grid(row=0, column=3, padx=5, pady=5)

        ttk.Label(frame, text="Allocation (%):").grid(row=0, column=4, padx=5, pady=5)
        self.allocation_var = tk.DoubleVar(value=0.0)
        ttk.Entry(frame, textvariable=self.allocation_var, width=10).grid(row=0, column=5, padx=5, pady=5)

        self.add_button = ttk.Button(frame, text="Add", command=self.add_asset)
        self.add_button.grid(row=0, column=6, padx=10, pady=5)
        ttk.Button(frame, text="Reset", command=self.reset_portfolio).grid(row=0, column=7, padx=10, pady=5)

        # Treeview for portfolio display
        columns = ("Source", "Ticker", "Allocation")
        self.tree = ttk.Treeview(frame, columns=columns, show="headings", height=8)
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=120)
        self.tree.grid(row=1, column=0, columnspan=8, pady=10)

        # Allocation display
        self.alloc_label = ttk.Label(frame, text="Total Allocation: 0.0%", font=("Arial", 10, "bold"))
        self.alloc_label.grid(row=2, column=0, columnspan=8, pady=5)

    def add_asset(self):
        source = self.source_var.get()
        ticker = self.ticker_var.get().strip().upper()
        allocation = self.allocation_var.get()

        if not ticker or allocation <= 0:
            messagebox.showwarning("Invalid Input", "Please enter a valid ticker and positive allocation.")
            return

        total_alloc = sum(item["Allocation"] for item in self.portfolio) + allocation
        if total_alloc > 100:
            messagebox.showerror(
                "Allocation Limit Exceeded",
                f"Total allocation would be {total_alloc:.1f}%, exceeding 100%. Adjust allocations first.",
            )
            return

        self.portfolio.append({"Source": source, "Ticker": ticker, "Allocation": allocation})
        self.tree.insert("", "end", values=(source, ticker, f"{allocation:.1f}%"))
        self._update_total_allocation()

    def reset_portfolio(self):
        self.portfolio.clear()
        for row in self.tree.get_children():
            self.tree.delete(row)
        self._update_total_allocation()
        messagebox.showinfo("Portfolio Reset", "Portfolio cleared successfully.")

    def _update_total_allocation(self):
        total = sum(item["Allocation"] for item in self.portfolio)
        self.alloc_label.config(
            text=f"Total Allocation: {total:.1f}%",
            foreground=("red" if total > 100 else "green"),
        )
        self.add_button.config(state=("disabled" if total >= 100 else "normal"))

    # =============================
    # Leverage & Interest Section
    # =============================
    def _build_leverage_section(self):
        frame = ttk.LabelFrame(self.root, text="Leverage and Interest Settings", padding=10)
        frame.pack(fill="x", padx=10, pady=10)

        self.use_leverage = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Apply Leverage & Interest Rate", variable=self.use_leverage, command=self._toggle_leverage).grid(
            row=0, column=0, sticky="w", padx=5, pady=5
        )

        ttk.Label(frame, text="Leverage (x):").grid(row=0, column=1, padx=5)
        self.leverage_var = tk.DoubleVar(value=1.0)
        self.leverage_entry = ttk.Entry(frame, textvariable=self.leverage_var, width=10, state="disabled")
        self.leverage_entry.grid(row=0, column=2, padx=5)

        ttk.Label(frame, text="Borrow Rate (%):").grid(row=0, column=3, padx=5)
        self.interest_var = tk.DoubleVar(value=0.0)
        self.interest_entry = ttk.Entry(frame, textvariable=self.interest_var, width=10, state="disabled")
        self.interest_entry.grid(row=0, column=4, padx=5)

    def _toggle_leverage(self):
        state = "normal" if self.use_leverage.get() else "disabled"
        self.leverage_entry.config(state=state)
        self.interest_entry.config(state=state)

        if not self.use_leverage.get():
            self.leverage_var.set(1.0)
            self.interest_var.set(0.0)

    # =============================
    # Strategy Section
    # =============================
    def _build_strategy_section(self):
        frame = ttk.LabelFrame(self.root, text="Withdrawal Strategy", padding=10)
        frame.pack(fill="x", padx=10, pady=10)

        ttk.Label(frame, text="Strategy:").grid(row=0, column=0, padx=5, pady=5)
        self.strategy_var = tk.StringVar(value="None")
        self.strategy_box = ttk.Combobox(
            frame,
            textvariable=self.strategy_var,
            values=["None", "Fixed Percentage", "Dynamic Rule-Based", "Guyton-Klinger"],
            width=25,
        )
        self.strategy_box.grid(row=0, column=1, padx=5, pady=5)
        self.strategy_box.bind("<<ComboboxSelected>>", self._on_strategy_select)

        ttk.Button(frame, text="Expand Strategy", command=self._expand_strategy).grid(row=0, column=2, padx=10)

        self.param_frame = ttk.Frame(frame)
        self.param_frame.grid(row=1, column=0, columnspan=4, sticky="w", pady=10)

    def _on_strategy_select(self, event):
        for widget in self.param_frame.winfo_children():
            widget.destroy()

        strategy = self.strategy_var.get()
        if strategy == "Fixed Percentage":
            ttk.Label(self.param_frame, text="Withdrawal Rate (%):").grid(row=0, column=0, padx=5)
            self.rate_var = tk.DoubleVar()
            ttk.Entry(self.param_frame, textvariable=self.rate_var, width=10).grid(row=0, column=1, padx=5)

        elif strategy == "Dynamic Rule-Based":
            ttk.Label(self.param_frame, text="Min Rate (%):").grid(row=0, column=0, padx=5)
            self.min_rate_var = tk.DoubleVar()
            ttk.Entry(self.param_frame, textvariable=self.min_rate_var, width=10).grid(row=0, column=1, padx=5)

            ttk.Label(self.param_frame, text="Max Rate (%):").grid(row=0, column=2, padx=5)
            self.max_rate_var = tk.DoubleVar()
            ttk.Entry(self.param_frame, textvariable=self.max_rate_var, width=10).grid(row=0, column=3, padx=5)

        elif strategy == "Guyton-Klinger":
            ttk.Label(self.param_frame, text="Base Rate (%):").grid(row=0, column=0, padx=5)
            self.base_rate_var = tk.DoubleVar()
            ttk.Entry(self.param_frame, textvariable=self.base_rate_var, width=10).grid(row=0, column=1, padx=5)

            ttk.Label(self.param_frame, text="Guardrails (%):").grid(row=0, column=2, padx=5)
            self.guardrails_var = tk.DoubleVar()
            ttk.Entry(self.param_frame, textvariable=self.guardrails_var, width=10).grid(row=0, column=3, padx=5)

    # =============================
    # Strategy Expansion Placeholder
    # =============================
    def _expand_strategy(self):
        """Example expansion function: add covered calls, puts, etc."""
        win = tk.Toplevel(self.root)
        win.title("Strategy Expansion")
        win.geometry("400x300")

        ttk.Label(win, text="Advanced Strategy Options", font=("Arial", 12, "bold")).pack(pady=10)
        ttk.Label(win, text="Here you can later define advanced tactics like:").pack(pady=5)
        ttk.Label(win, text="• Covered Calls\n• Protective Puts\n• Option Spreads\n• Dynamic Hedging\n• Custom Rules", justify="left").pack(pady=5)

        ttk.Button(win, text="Close", command=win.destroy).pack(pady=15)

    # =============================
    # Action Section
    # =============================
    def _build_action_section(self):
        frame = ttk.LabelFrame(self.root, text="Actions", padding=10)
        frame.pack(fill="x", padx=10, pady=10)

        ttk.Button(frame, text="Analyse", command=self.analyse_portfolio).grid(row=0, column=0, padx=10)
        ttk.Button(frame, text="Run Monte Carlo", command=self.run_monte_carlo).grid(row=0, column=1, padx=10)
        ttk.Button(frame, text="Markowitz", command=self.run_markowitz).grid(row=0, column=2, padx=10)

    # =============================
    # Placeholder Action Functions
    # =============================
    def analyse_portfolio(self):
        messagebox.showinfo("Analysis", "Portfolio analysis will be added later.")

    def run_monte_carlo(self):
        messagebox.showinfo("Monte Carlo", "Monte Carlo simulation placeholder.")

    def run_markowitz(self):
        messagebox.showinfo("Markowitz", "Markowitz optimization placeholder.")


if __name__ == "__main__":
    root = tk.Tk()
    app = PortfolioGUI(root)
    root.mainloop()

