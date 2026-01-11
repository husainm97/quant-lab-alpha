# 📊 quant-lab-alpha — Portfolio Factor Regression & Analysis Toolkit ![Python Version](https://img.shields.io/badge/python-3.12%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green)

**Quant Lab Alpha** is a Python toolkit designed to quantify **where portfolio risk comes from, how it evolves over time, and the statistical distribution of outcomes**.

It bridges the gap between static factor analysis and long-horizon strategy modelling by connecting **portfolio construction**, **factor and asset-level risk decomposition**, and **outcome-oriented Monte Carlo simulations** under consistent assumptions. The result is a transparent workflow that makes it easier to understand not just *what* a portfolio is exposed to, but *why* those exposures matter over time.

The framework is built around the **Fama–French Five-Factor (FF5) Model**, with extensions into **rolling factor regressions**, **block bootstrap simulations**, and **stabilised Markowitz portfolio optimisation** to visualise both risk structure and long-term outcomes under market-like and stressed conditions.

📓 Methodological explanations, validation notebooks, and visual diagnostics are provided under `notebooks/`.  
📬 For questions, comments, or bug reports, contact **husainm97@gmail.com**.

---

## ⚠️ Disclaimer

**This repository is intended for quantifying risk exposures and statistical outcomes.**  
All calculations, regressions, and results are **not financial advice**.  
Past performance does **not guarantee future results**.

Any example portfolios included in this project are **approximations of conceived factor exposure strategies**, not investment recommendations.  
Users are solely responsible for any decisions or applications derived from this toolkit.

---

## 🙏 Acknowledgements

The developer thanks **Curvo.eu** and **Benjamin Felix (PWL Capital, Canada)** for permission to use selected datasets and published results.  
These have been used solely to benchmark and validate analysis code within the Jupyter notebooks.

---

## 🖥️ Interactive GUI (Tkinter)  
**Implemented**

![GUI](images/v1-0-1/GUI_Light.png)

- Simplified GUI-based portfolio builder  
- Save and import portfolios and settings  
- Toggle leveraged strategies  
- One-click access to the full analysis suite  

---

## 1️⃣ Fama–French Five-Factor (FF5) Regressions  
**Implemented**

![Factor Regression](images/v-1-0-1/Factor_Regression_Light.png)  
![Rolling Regressions](images/v1-0-1/Rolling_Regressions_Light.png)

- Factor data ingestion from the Ken French Data Library  
- Portfolio- and asset-level risk factor regressions  
- **Rolling regressions** over configurable windows (e.g. 3y / 5y / 10y)  
- Factor exposure, alpha, and contribution analysis  

---

## 2️⃣ Markowitz Portfolio Optimiser  
**Implemented**

![Markowitz Optimisation](images/v1-0-1/Markowitz_Light.png)

- Mean–variance optimisation  
- Covariance estimation from historical or factor-implied returns  
- Efficient frontier construction  
- **Ledoit–Wolf covariance matrix shrinkage**  
- Direct application of optimal weights to the active portfolio  

---

## 3️⃣ Risk Report  
**Implemented**

![Risk Report](images/v1-0-0/Risk_Report.png)

- Historical drawdown analysis  
- Asset- and factor-level risk contribution assessment  
- Monthly CVaR and VaR at 95% confidence  

---

## 4️⃣ Correlation Matrix  
**Implemented**

![Correlation Matrix](images/v1-0-0/Correlation_Matrix.png)

- Inter-asset correlation inspection  
- Heatmap visualisation for rapid structure assessment  

---

## 5️⃣ Monte Carlo Retirement Simulation  
**Implemented**

![Monte Carlo Simulation](images/v1-0-1/Monte_Carlo_Light.png)
![Monte Carlo Simulation](images/v1-0-1/Monte_Carlo_Diag_Light.png)

- Synthetic return histories generated via FF5-fitted models  
- Block / bootstrap-style simulations  
- Stress tests: return shift, volatility shocks  
- Multiple withdrawal strategies:
  - Fixed 4% (initial capital)  
  - Variable 4% of current capital  
  - Guardrails (2.5–5%)  
  - Bucket strategy  
- Failure probability and terminal wealth analysis  

**In Development**
- Improved stress testing and scenario labelling  
- Plug-and-play withdrawal strategy implementation  

---

## 💱 Currency Support  
**Implemented**

- Automatic currency detection via Yahoo Finance metadata  
- FX normalisation to USD for cross-currency portfolios  
- Seamless integration into FF5 analysis and Monte Carlo simulations  

---

## ⚖️ Assumptions and Limitations

To maintain interpretability and analytical clarity, several regional, institutional, and broker-dependent real-world effects are intentionally excluded:

Key limitations include:

- Available currencies are currently limited to 9 options: "USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "CNY", "INR"
- Portfolio weights are assumed to remain constant (rebalancing planned)  
- Transaction costs are not modelled  
- Leverage multipliers are static; real-world borrowing constraints are not modelled  
- Assets with short histories may be extended using FF5-based synthetic returns  
- Taxes are **not** modelled; withdrawals represent spending *plus* taxes  
- The Markowitz optimiser uses randomised sampling; direct solvers (e.g. `pypfopt`) would be more robust  
- Market data is sourced from Yahoo Finance; additional providers are planned  

This simplifies interpretation but may deviate from realisable outcomes.

---

## 🧪 Realism Update (In Development)

Features that materially increase realism but are currently separated from the core engine:

- **Portfolio rebalancing**  
- <del>**Inflation-adjusted withdrawals**</del> [Done ✅]
- Limited leverage model with loan-to-value caps and margin calls  
- Real (not just nominal) return tracking  

These will be introduced as **opt-in layers**, not hardwired assumptions.

---

## 🐞 Known Issues

- Please check out the Issues tab to open an issue or contact the developer to report a bug.  
- Recent fixes:
  - Markowitz optimiser weights now correctly apply in the Treeview  
  - Negative wealth paths are terminated in Monte Carlo simulations  
  - Leverage application before simulations fixed  

---

Pull requests, suggestions and bug reports are welcome.

---

## 🚀 Getting Started

### 1. Clone the repository
```
bash
git clone https://github.com/husainm97/quant-lab-alpha.git
cd quant-lab-alpha
```

### 2. **Install dependencies** 
```
pip install -r requirements.txt
```

### 3. **Launch the interface**
```
python main.py
```
## 🛠️ Prerequisites & Troubleshooting

This project uses **Tkinter** for the graphical user interface. Many Linux distributions (like Ubuntu and Fedora) package Python modularly and do not include Tkinter by default.

If you encounter `ModuleNotFoundError: No module named 'tkinter'`, run the command for your specific operating system:

### 🐧 Linux Setup
* **Ubuntu / Debian / Mint / Kali:**
    ```bash
    sudo apt update
    sudo apt install python3-tk
    ```
* **Fedora / RHEL / CentOS:**
    ```bash
    sudo dnf install python3-tkinter
    ```
* **Arch Linux:**
    ```bash
    sudo pacman -S tk
    ```
