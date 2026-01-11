#!/usr/bin/env python3
# gui.py — Quant Lab Alpha basic GUI for factor analysis

"""
This file currently contains the main GUI implementation for Quant Lab Alpha. 
Analysis suite currently offers Fama-French Factor Regressions, Markowitz Optimisation, Correlation Matrix, Risk Report (CVaR, drawdown) and Monte Carlo Simulations.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from simulations.portfolio_module import Portfolio
from data.fetcher import fetch
import json
from tkinter import filedialog
import sv_ttk


class PortfolioGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Quant Lab Alpha")
        self.root.geometry("850x750")

        self.root.update_idletasks()
        self.root.minsize(850, 750)
        self.root.resizable(True, True)

        sv_ttk.set_theme("light") # Start with light
        self.is_dark = False

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
        self._build_config_section()
        self._build_action_section()

    def toggle_theme(self):
        if sv_ttk.get_theme() == "dark":
            sv_ttk.set_theme("light")
            self.is_dark=False
        else:
            sv_ttk.set_theme("dark")
            self.is_dark=True


    # =============================
    # Portfolio Section
    # =============================
    def _build_portfolio_section(self):
        frame = ttk.LabelFrame(self.root, text="Portfolio Builder", padding=10)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        input_wrapper = ttk.Frame(frame)
        input_wrapper.grid(row=0, column=0, columnspan=8, sticky="w")

        ttk.Label(input_wrapper, text="Source:").pack(side="left", padx=2)
        self.source_var = tk.StringVar(value="Yahoo")
        ttk.Combobox(input_wrapper, textvariable=self.source_var, values=["Yahoo", "Source", "Manual"], width=10).pack(side="left", padx=5)

        ttk.Label(input_wrapper, text="Ticker:").pack(side="left", padx=2)
        self.ticker_var = tk.StringVar()
        ttk.Entry(input_wrapper, textvariable=self.ticker_var, width=12).pack(side="left", padx=5)

        ttk.Label(input_wrapper, text="Alloc (%):").pack(side="left", padx=2)
        self.allocation_var = tk.DoubleVar(value=0.0)
        ttk.Entry(input_wrapper, textvariable=self.allocation_var, width=6).pack(side="left", padx=5)

        self.add_button = ttk.Button(frame, text="Add", command=self.add_asset)
        self.add_button.grid(row=0, column=6, padx=10, pady=5)
        self.delete_button = ttk.Button(frame, text="Delete", command=self.delete_asset)
        self.delete_button.grid(row=0, column=7, padx=10, pady=5)

        # --- Save/Load Portfolio ---
        file_frame = ttk.Frame(frame)
        file_frame.grid(row=3, column=0, columnspan=8, pady=5)

        ttk.Button(file_frame, text="Reset", command=self.reset_portfolio).pack(side="left", padx=5)
        ttk.Button(file_frame, text="Import JSON", command=self.import_portfolio).pack(side="left", padx=5)
        ttk.Button(file_frame, text="Save Portfolio", command=self.save_portfolio).pack(side="left", padx=5)

        # Treeview for portfolio display
        columns = ("Source", "Ticker", "Allocation")
        self.tree = ttk.Treeview(frame, columns=columns, show="headings", height=8)
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=180)
        self.tree.grid(row=1, column=0, columnspan=8, pady=10, sticky="nsew")
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)

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
                # Placeholder for future extensions
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
            
            # 3. Create progress window
            assets = data.get("assets", [])
            progress_win = tk.Toplevel(self.root)
            progress_win.title("Loading Portfolio")
            progress_win.geometry("400x100")
            progress_win.transient(self.root)
            progress_win.grab_set()
            
            ttk.Label(progress_win, text="Fetching asset data...", font=("Arial", 10)).pack(pady=10)
            
            progress_bar = ttk.Progressbar(progress_win, length=350, mode='determinate', maximum=len(assets))
            progress_bar.pack(pady=10)
            
            status_label = ttk.Label(progress_win, text="", font=("Arial", 9))
            status_label.pack()
            
            # 4. Batch Add Assets with progress
            for i, asset in enumerate(assets):
                status_label.config(text=f"Loading {asset['Ticker']} ({i+1}/{len(assets)})")
                progress_win.update()
                
                self.source_var.set(asset["Source"])
                self.ticker_var.set(asset["Ticker"])
                self.allocation_var.set(asset["Allocation"])
                self.add_asset()
                
                progress_bar['value'] = i + 1
                progress_win.update()
            
            progress_win.destroy()
            messagebox.showinfo("Import Success", f"Loaded {len(assets)} assets successfully!")
            
        except Exception as e:
            if 'progress_win' in locals():
                progress_win.destroy()
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
    

    def _build_config_section(self):
            """Build the configuration/settings section with auto-updates and info buttons"""
            frame = ttk.LabelFrame(self.root, text="Configuration & Settings", padding=10)
            frame.pack(fill="x", padx=10, pady=10)
            
            # --- Currency Settings ---
            ttk.Label(frame, text="Base Currency:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
            
            self.currency_var = tk.StringVar(value="USD")
            currency_combo = ttk.Combobox(
                frame,
                textvariable=self.currency_var,
                values=["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "CNY", "INR"],
                width=8,
                state="readonly"
            )
            currency_combo.grid(row=0, column=1, sticky="w", padx=5, pady=5)
            
            # Restore Bind
            currency_combo.bind("<<ComboboxSelected>>", lambda e: self._apply_configuration())
            
            # Restore Currency info button
            ttk.Button(frame, text="?", width=3, 
                    command=self._show_currency_info).grid(row=0, column=2, padx=5, pady=5)
            
            # --- Factor Region Settings ---
            ttk.Label(frame, text="Factor Region:").grid(row=0, column=3, sticky="w", padx=(20, 5), pady=5)
            
            self.ff_region_var = tk.StringVar(value="Developed")
            region_combo = ttk.Combobox(
                frame,
                textvariable=self.ff_region_var,
                values=["Developed", "US", "Developed ex-US", "Japan", "Emerging Markets", "Asia-Pacific ex-Japan"],
                width=20,
                state="readonly"
            )
            region_combo.grid(row=0, column=4, sticky="w", padx=5, pady=5)
            
            # Restore Bind
            region_combo.bind("<<ComboboxSelected>>", lambda e: self._apply_configuration())
            
            # Restore Region info button
            ttk.Button(frame, text="?", width=3, 
                    command=self._show_region_info).grid(row=0, column=5, padx=5, pady=5)
            
            # Add the toggle button
            self.theme_button = ttk.Button(frame, text="Light/Dark", command=self.toggle_theme)
            self.theme_button.grid(row=0, column=6, padx=10, pady=5)


    def _show_currency_info(self):
        """Display information about currency conversion"""
        info_win = tk.Toplevel(self.root)
        info_win.title("Currency Settings")
        info_win.geometry("500x350")
        info_win.resizable(False, False)
        
        main_frame = ttk.Frame(info_win, padding=15)
        main_frame.pack(fill="both", expand=True)
        
        # Title
        ttk.Label(
            main_frame, 
            text="Base Currency Settings",
            font=("Segoe UI", 16, "bold")
        ).pack(pady=(0, 10))
        
        # Description
        desc_text = (
            "Select your preferred base currency for portfolio analysis.\n\n"
            "All asset prices and returns will be converted to this currency "
            "for consistent comparison and analysis.\n"
            "Fama-French Factor regressions are calculated in USD.\n\n"
            "Supported currencies:\n"
            "• USD - United States Dollar\n"
            "• EUR - Euro\n"
            "• GBP - British Pound Sterling\n"
            "• JPY - Japanese Yen\n"
            "• CHF - Swiss Franc\n"
            "• CAD - Canadian Dollar\n"
            "• AUD - Australian Dollar\n"
            "• CNY - Chinese Yuan\n"
            "• INR - Indian Rupee\n\n"
            "Note: Currency conversion uses historical exchange rates "
            "to ensure accurate return calculations."
        )
        
        ttk.Label(
            main_frame, 
            text=desc_text, 
            justify="left",
            wraplength=450
        ).pack(pady=(0, 15))
        
        # Close button
        ttk.Button(
            main_frame, 
            text="Close", 
            command=info_win.destroy
        ).pack(pady=(10, 0))


    def _show_region_info(self):
        """Display information about Fama-French regional datasets in a spacious window"""
        info_win = tk.Toplevel(self.root)
        info_win.title("Fama-French Regional Datasets")
        
        # 1. Expand the window size so no scrolling is needed
        info_win.geometry("850x650") 
        info_win.resizable(True, True)
        
        main_frame = ttk.Frame(info_win, padding=25)
        main_frame.pack(fill="both", expand=True)
        
        # Header
        ttk.Label(
            main_frame, 
            text="Fama-French 5-Factor Regional Datasets",
            font=("Segoe UI", 16, "bold")
        ).pack(pady=(0, 20), anchor="w")
        
        # 2. Text widget configured to look like a document, not a "box"
        region_text = tk.Text(
            main_frame,
            wrap="word",
            font=("Segoe UI", 10),
            bg=info_win.cget("bg"), # Match system background
            relief="flat",          # Remove borders
            highlightthickness=0,   # Remove focus ring
            cursor="arrow"          # Keep it feeling like an info page
        )
        region_text.pack(fill="both", expand=True)

        # 3. Precision Alignment Tags
        # spacing1: space before paragraph | spacing2: space between wrapped lines
        region_text.tag_configure("header", font=("Segoe UI", 11, "bold"), spacing1=15, spacing3=2)
        region_text.tag_configure("body", lmargin1=0, lmargin2=0, spacing2=3)
        region_text.tag_configure("source", font=("Segoe UI", 9, "italic"), foreground="gray", spacing1=30)

        # Content structure
        content = [
            ("DEVELOPED MARKETS", "Includes 23 countries: Australia, Austria, Belgium, Canada, Switzerland, Germany, Denmark, Spain, Finland, France, Great Britain, Greece, Hong Kong, Ireland, Italy, Japan, Netherlands, Norway, New Zealand, Portugal, Sweden, Singapore, and United States. \nBest for: Global developed market exposure."),
            ("UNITED STATES", "US market only. \nBest for: US-focused portfolios and domestic factor analysis."),
            ("DEVELOPED EX-US", "All developed markets excluding the United States. \nBest for: International diversification and non-US developed exposure."),
            ("JAPAN", "Japanese market only. \nBest for: Japan-specific analysis and Asian developed market focus."),
            ("EMERGING MARKETS", "Includes major emerging economies across Asia, Latin America, Eastern Europe, and Africa. \nBest for: Emerging market exposure and growth-focused portfolios."),
            ("ASIA PACIFIC EX-JAPAN", "Developed and emerging markets in Asia Pacific excluding Japan (Hong Kong, Singapore, Australia, New Zealand, Korea, Taiwan, etc.). \nBest for: Asian regional focus without Japan exposure.")
        ]

        # Insert logic
        for title, desc in content:
            region_text.insert("end", f"{title}\n", "header")
            region_text.insert("end", f"{desc}\n", "body")
            
        region_text.insert("end", "\nSource: Kenneth R. French Data Library", "source")
        
        region_text.config(state="disabled")

        # Close button at the bottom right
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill="x", pady=(10, 0))
        ttk.Button(btn_frame, text="Close", command=info_win.destroy).pack(side="right")


    def _apply_configuration(self):
        """Apply the selected configuration settings"""
        currency = self.currency_var.get()
        region = self.ff_region_var.get()
        
        # Store in portfolio object or global config
        if hasattr(self, 'portfolio_obj'):
            self.portfolio_obj.base_currency = currency
            self.portfolio_obj.factor_region = region
        



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
            run_ff5_analysis(self.root, self.portfolio_obj, self.is_dark)
        except Exception as e:
            messagebox.showerror("Error", f"Regression failed:\n{e}")

    
    def run_markowitz(self):
        try:
            from simulations.markowitz import plot_markowitz_gui

            if not hasattr(self, "portfolio_obj") or not self.portfolio_obj.constituents:
                raise ValueError("Portfolio has no assets. Add assets first.")

            top = tk.Toplevel(self.root)
            top.title("Markowitz Efficient Frontier")
            plot_markowitz_gui(top, self.portfolio_obj, risk_free=0.02, gui=self, is_dark=self.is_dark)

        except Exception as e:
            messagebox.showerror("Error", f"Markowitz optimisation failed:\n{e}")

    def run_correlations(self):
        try:
            from src.correlation import plot_correlation_heatmap

            if not hasattr(self, "portfolio_obj") or not self.portfolio_obj.constituents:
                raise ValueError("Portfolio has no assets.")

            top = tk.Toplevel(self.root)
            top.title("Asset Correlation Matrix")

            plot_correlation_heatmap(top, self.portfolio_obj, is_dark=self.is_dark)

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
            plot_risk_dashboard(top, self.portfolio_obj, is_dark=self.is_dark)
        except Exception as e:
            messagebox.showerror("Error", f"Risk assessment failed:\n{e}")


    def run_monte_carlo(self):
        try:
            from simulations.bootstrap import run_simulation
            if not self.portfolio_obj.constituents:
                messagebox.showwarning("No Assets", "Add assets to the portfolio before running Monte Carlo.")
                return

            top = tk.Toplevel(self.root)
            top.title("Block Bootstrap Simulation")
            lev, rate = self.get_portfolio_leverage_settings()
            self.portfolio_obj.apply_leverage(lev, rate)
            run_simulation(top, portfolio_obj=self.portfolio_obj, is_dark=self.is_dark)

        # ... inside your function ...
        except Exception as e:
            #import traceback
            # This captures the full multi-line error log
            #full_error = traceback.format_exc()
            #print(full_error) # Prints to your terminal/console
            #messagebox.showerror("Detailed Error", f"Monte Carlo failed:\n\n{full_error}")
            messagebox.showerror("Error", f"Monte Carlo simulation failed:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PortfolioGUI(root)
    root.mainloop()