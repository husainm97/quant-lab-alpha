#!/usr/bin/env python3
# gui.py — Quant Lab Alpha basic GUI for factor analysis

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------
# Placeholder analysis function
# ----------------------------------------------------------
def run_factor_analysis(etf, model, window):
    # In your real implementation, this should:
    #   - Fetch data for selected ETF
    #   - Run regression against Fama–French factors
    #   - Plot results

    # Mock data for now:
    factors = ["MKT", "SMB", "HML", "RMW", "CMA"]
    loadings = np.random.randn(len(factors))

    plt.figure(figsize=(6, 4))
    plt.bar(factors, loadings, color="skyblue", edgecolor="black")
    plt.title(f"Factor Loadings for {etf} ({model}, window={window})")
    plt.xlabel("Factors")
    plt.ylabel("Loading")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------
# GUI setup
# ----------------------------------------------------------
def main():
    root = tk.Tk()
    root.title("Quant Lab Alpha — Factor Analysis")
    root.geometry("420x300")
    root.resizable(False, False)

    ttk.Label(root, text="Quant Lab Alpha", font=("Helvetica", 16, "bold")).pack(pady=10)
    ttk.Separator(root).pack(fill="x", padx=10, pady=5)

    # --- ETF selection ---
    etf_frame = ttk.Frame(root)
    etf_frame.pack(pady=10)
    ttk.Label(etf_frame, text="Select ETF:").grid(row=0, column=0, padx=5, sticky="w")
    etf_options = ["SPY", "EFA", "IWM", "VNQ", "QQQ", "GLD"]
    etf_var = tk.StringVar(value=etf_options[0])
    ttk.Combobox(etf_frame, textvariable=etf_var, values=etf_options, width=15, state="readonly").grid(row=0, column=1)

    # --- Model selection ---
    model_frame = ttk.Frame(root)
    model_frame.pack(pady=5)
    ttk.Label(model_frame, text="Model:").grid(row=0, column=0, padx=5, sticky="w")
    model_options = ["Fama–French 3-Factor", "Fama–French 5-Factor", "Custom Factors"]
    model_var = tk.StringVar(value=model_options[1])
    ttk.Combobox(model_frame, textvariable=model_var, values=model_options, width=25, state="readonly").grid(row=0, column=1)

    # --- Rolling window ---
    window_frame = ttk.Frame(root)
    window_frame.pack(pady=5)
    ttk.Label(window_frame, text="Rolling window (days):").grid(row=0, column=0, padx=5, sticky="w")
    window_options = ["30", "60", "90", "180"]
    window_var = tk.StringVar(value=window_options[2])
    ttk.Combobox(window_frame, textvariable=window_var, values=window_options, width=10, state="readonly").grid(row=0, column=1)

    # --- Run button ---
    def on_run():
        etf = etf_var.get()
        model = model_var.get()
        window = window_var.get()
        messagebox.showinfo("Running Analysis", f"Generating factor plots for {etf}...")
        run_factor_analysis(etf, model, window)

    ttk.Button(root, text="Run Analysis", command=on_run).pack(pady=20)

    ttk.Label(root, text="© 2025 Quant Lab Alpha", font=("Helvetica", 8)).pack(side="bottom", pady=5)

    root.mainloop()


if __name__ == "__main__":
    main()
