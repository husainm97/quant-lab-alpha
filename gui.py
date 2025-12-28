#!/usr/bin/env python3
# gui.py — Quant Lab Alpha basic GUI for factor analysis

"""
This file currently contains a basic tkinter GUI that allows building a portfolio from various sources and choosing leverage and withdrawal strategies. 
Options for Markowitz optimisation, 5 factor analysis and retirement spending simulations are available. We are still to plug the right functions to the buttons!
"""

import tkinter as tk
from tkinter import ttk, messagebox
from simulations.portfolio_module import Portfolio
from data.fetcher import fetch
import json
from tkinter import filedialog


class PortfolioGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Portfolio Builder")
        self.root.geometry("720x550")

        self.portfolio = []
        self.strategy_params = {}
        self.leverage = 1.0
        self.interest_rate = 0.0
        self.portfolio_obj = Portfolio()

        self._build_ui()

    # =============================
    # Main UI Builder
    # =============================
    def _build_ui(self):
        self._build_portfolio_section()
        self._build_leverage_section()
        #self._build_strategy_section()
        self._build_action_section()

    # =============================
    # Portfolio Section
    # =============================
    def _build_portfolio_section(self):
        frame = ttk.LabelFrame(self.root, text="Select Assets", padding=10)
        frame.pack(fill="x", padx=10, pady=10)

        ttk.Label(frame, text="Source:").grid(row=0, column=0, padx=5, pady=5)
        self.source_var = tk.StringVar(value="Yahoo")
        ttk.Combobox(frame, textvariable=self.source_var, values=["Yahoo", "Source", "Manual"], width=10).grid(
            row=0, column=1, padx=5, pady=5
        )

        ttk.Label(frame, text="Ticker:").grid(row=0, column=2, padx=5, pady=5)
        self.ticker_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.ticker_var, width=12).grid(row=0, column=3, padx=5, pady=5)

        ttk.Label(frame, text="Allocation (%):").grid(row=0, column=4, padx=5, pady=5)
        self.allocation_var = tk.DoubleVar(value=0.0)
        ttk.Entry(frame, textvariable=self.allocation_var, width=6).grid(row=0, column=5, padx=5, pady=5)

        self.add_button = ttk.Button(frame, text="Add", command=self.add_asset)
        self.add_button.grid(row=0, column=6, padx=10, pady=5)
        self.delete_button = ttk.Button(frame, text="Delete", command=self.delete_asset)
        self.delete_button.grid(row=0, column=7, padx=10, pady=5)

        # --- Save/Load Portfolio ---
        file_frame = ttk.Frame(frame)
        file_frame.grid(row=3, column=0, columnspan=9, pady=5)

        ttk.Button(file_frame, text="Reset", command=self.reset_portfolio).pack(side="left", padx=5)
        ttk.Button(file_frame, text="Import JSON", command=self.import_portfolio).pack(side="left", padx=5)
        ttk.Button(file_frame, text="Save Portfolio", command=self.save_portfolio).pack(side="left", padx=5)

        # Treeview for portfolio display
        columns = ("Source", "Ticker", "Allocation")
        self.tree = ttk.Treeview(frame, columns=columns, show="headings", height=8)
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=180)
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

        # Fetch data for the ticker
        try:
            if source.lower() == "yahoo":
                df = fetch(f"Yahoo/{ticker}")
            else:
                # You can extend this to other sources if you define them in your registry
                df = fetch(f"{source}/{ticker}")

            if df.empty:
                messagebox.showwarning("Data Warning", f"No data returned for {ticker} from {source}.")
                return

        except Exception as e:
            messagebox.showerror("Fetch Error", f"Failed to fetch {ticker} from {source}:\n{e}")
            return

        # Add asset with data to portfolio
        self.portfolio.append({"Source": source, "Ticker": ticker, "Allocation": allocation, "Data": df})
        self.portfolio_obj.add_asset(ticker, allocation / 100, df=df)
        self.tree.insert("", "end", values=(source, ticker, f"{allocation:.1f}%"))
        self._update_total_allocation()

    def delete_asset(self):
        # Get the selected item from the Treeview
        selected_items = self.tree.selection()
        
        if not selected_items:
            messagebox.showwarning("Delete Error", "Please select an asset from the list to delete.")
            return

        for item in selected_items:
            # Identify the ticker from the selected row
            values = self.tree.item(item, "values")
            ticker_to_remove = values[1]

            # Remove from internal portfolio list
            self.portfolio = [p for p in self.portfolio if p["Ticker"] != ticker_to_remove]

            # Remove from your portfolio object (if it supports removal)
            # If your portfolio_obj doesn't have a remove method, you may need to 
            # rebuild it or add a remove_asset method to its class.
            if hasattr(self.portfolio_obj, 'remove_asset'):
                self.portfolio_obj.remove_asset(ticker_to_remove)
            # Remove from the Treeview UI
            self.tree.delete(item)

        # Update the total allocation label
        self._update_total_allocation()
        messagebox.showinfo("Deleted", f"Removed {len(selected_items)} asset(s) from portfolio.")

    def reset_portfolio(self):
        self.portfolio.clear()
        for row in self.tree.get_children():
            self.tree.delete(row)
        self._update_total_allocation()
        self.portfolio_obj.reset()

    def _update_total_allocation(self):
        total = sum(item["Allocation"] for item in self.portfolio)
        self.alloc_label.config(
            text=f"Total Allocation: {total:.1f}%",
            foreground=("red" if total > 100 else "green"),
        )
        self.add_button.config(state=("disabled" if total >= 100 else "normal"))

    def save_portfolio(self):
        # Prepare the data dictionary
        data = {
            "leverage": self.leverage_var.get(), # Assuming you have these variables
            "interest_rate": self.interest_var.get(),
            "assets": [
                {
                    "Source": item["Source"],
                    "Ticker": item["Ticker"],
                    "Allocation": item["Allocation"]
                } 
                for item in self.portfolio
            ]
        }

        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)
            messagebox.showinfo("Success", "Portfolio saved successfully!")

    def import_portfolio(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not file_path:
            return

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # 1. Clear current portfolio
            self.reset_portfolio()

            # 2. Set global parameters
            self.leverage_var.set(data.get("leverage", 1.0))
            self.interest_var.set(data.get("interest_rate", 0.0))

            # 3. Batch Add Assets
            for asset in data.get("assets", []):
                # We set the UI variables so we can reuse your existing add_asset logic
                self.source_var.set(asset["Source"])
                self.ticker_var.set(asset["Ticker"])
                self.allocation_var.set(asset["Allocation"])
                self.add_asset() # Reusing your existing validation and fetch logic

            messagebox.showinfo("Import Success", "Portfolio loaded and data fetched.")
            
        except Exception as e:
            messagebox.showerror("Import Error", f"Failed to load file:\n{e}")

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
        # enable/disable entries
        state = "normal" if self.use_leverage.get() else "disabled"
        self.leverage_entry.config(state=state)
        self.interest_entry.config(state=state)

        # read current entry values and apply
        lev = max(1.0, self.leverage_var.get()) if self.use_leverage.get() else 1.0
        rate = max(0.0, self.interest_var.get()/100) if self.use_leverage.get() else 0.0

        try:
            self.portfolio_obj.apply_leverage(lev, rate)
            print(f"Applied leverage={self.portfolio_obj.leverage}, interest={self.portfolio_obj.interest_rate}")
        except ValueError as e:
            messagebox.showerror("Leverage Error", str(e))

    def get_portfolio_leverage_settings(self):
        """Return leverage & interest to use based on checkbox & entry fields."""
        if self.use_leverage.get():
            lev = max(1.0, self.leverage_var.get())
            rate = max(0.0, self.interest_var.get() / 100)
        else:
            lev = 1.0
            rate = 0.0
        return lev, rate


    '''
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
    '''

    # =============================
    # Action Section
    # =============================
    def _build_action_section(self):
        frame = ttk.LabelFrame(self.root, text="Actions", padding=10)
        frame.pack(fill="x", padx=10, pady=10)

        ttk.Button(frame, text="Factor Regression", command=self.analyse_portfolio).grid(row=0, column=0, padx=10)
        ttk.Button(frame, text="Markowitz Optimiser", command=self.run_markowitz).grid(row=0, column=1, padx=10)
        ttk.Button(frame, text="Correlation Matrix", command=self.run_correlations).grid(row=0, column=2, padx=10)
        ttk.Button(frame, text="Risk Report", command=self.run_risk_report).grid(row=0, column=3, padx=10)
        ttk.Button(frame, text="Run Monte Carlo", command=self.run_monte_carlo).grid(row=0, column=4, padx=10)
        

    # =============================
    # Placeholder Action Functions
    # =============================
    def analyse_portfolio(self):
        # Do not use until currency conversion is implemented!!!
        try:
            from src.five_factor_model import run_ff5_analysis

            if not self.portfolio_obj.constituents:
                messagebox.showwarning("No Assets", "Add assets to the portfolio before running Monte Carlo.")
                return

            lev, rate = self.get_portfolio_leverage_settings()
            self.portfolio_obj.apply_leverage(lev, rate)
            run_ff5_analysis(self.root, self.portfolio_obj)
        except Exception as e:
            messagebox.showerror("Error", f"Regression failed:\n{e}")

    
    def run_markowitz(self):
        try:
            from simulations.markowitz import plot_markowitz_gui

            if not hasattr(self, "portfolio_obj") or not self.portfolio_obj.constituents:
                raise ValueError("Portfolio has no assets. Add assets first.")

            top = tk.Toplevel(self.root)
            top.title("Markowitz Efficient Frontier")
            plot_markowitz_gui(top, self.portfolio_obj, risk_free=0.02, gui=self)

        except Exception as e:
            messagebox.showerror("Error", f"Markowitz optimisation failed:\n{e}")

    def run_correlations(self):
        try:
            from src.correlation import plot_correlation_heatmap

            if not hasattr(self, "portfolio_obj") or not self.portfolio_obj.constituents:
                raise ValueError("Portfolio has no assets.")

            top = tk.Toplevel(self.root)
            top.title("Asset Correlation Matrix")

            plot_correlation_heatmap(top, self.portfolio_obj)

        except Exception as e:
            import traceback
            full_error = traceback.format_exc()
            print(full_error)
            messagebox.showerror("Correlation Error", full_error)


    def run_risk_report(self):
        try:
            from src.risk import plot_risk_dashboard
            if not hasattr(self, "portfolio_obj") or not self.portfolio_obj.constituents:
                raise ValueError("Portfolio has no assets.")

            top = tk.Toplevel(self.root)
            top.title("Portfolio Risk Analytics")
            plot_risk_dashboard(top, self.portfolio_obj)
        except Exception as e:
            messagebox.showerror("Error", f"Risk assessment failed:\n{e}")


    def run_monte_carlo(self):
        try:
            from simulations.bootstrap import run_simulation
            if not self.portfolio_obj.constituents:
                messagebox.showwarning("No Assets", "Add assets to the portfolio before running Monte Carlo.")
                return

            lev, rate = self.get_portfolio_leverage_settings()
            self.portfolio_obj.apply_leverage(lev, rate)
            results, fig = run_simulation(portfolio_obj=self.portfolio_obj)


            # display results (same as before)
            summary = []
            for strat, res in results.items():
                line = (
                    f"{strat}:\n"
                    f"  Fail rate: {res['fail_rate']:.1f}%\n"
                    f"  Median final wealth: ${res['p50']:,.0f}\n"
                    f"  Chance to reach target: {res['prob_target']:.1f}%\n"
                )
                summary.append(line)
            
            messagebox.showinfo("Bootstrap Simulation Complete", "\n\n".join(summary))
            fig.show()

        #except Exception as e:
        #    messagebox.showerror("Error", f"Monte Carlo failed:\n{e}")

        # ... inside your function ...
        except Exception as e:
            import traceback
            # This captures the full multi-line error log
            full_error = traceback.format_exc()
            print(full_error) # Prints to your terminal/console
            messagebox.showerror("Detailed Error", f"Monte Carlo failed:\n\n{full_error}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PortfolioGUI(root)
    root.mainloop()

