# quant-lab-alpha — Portfolio Factor Regression & Analysis Toolkit

**Quant Lab Alpha** is a Python toolkit designed to quantify **where portfolio risk comes from, how it evolves over time, and the statistical distribution of outcomes**.

It bridges the gap between static factor analysis and long-horizon strategy modelling by connecting **portfolio construction**, **factor and asset-level risk decomposition**, and **outcome-oriented Monte Carlo simulations** under consistent assumptions. The result is a transparent workflow that makes it easier to understand not just *what* a portfolio is exposed to, but *why* those exposures matter over time.

The framework is built around the **Fama–French Five-Factor (FF5) Model**, with extensions into **rolling factor regressions**, **block bootstrap simulations**, and **stabilised Markowitz portfolio optimisation** to visualise both risk structure and long-term outcomes under market-like and stressed conditions.

Methodological explanations, validation notebooks, and visual diagnostics are provided under `notebooks/`.  
For questions, comments, or bug reports, contact **husainm97@gmail.com**.


---

## Disclaimer

**This repository is intended for quantifying risk exposures and statistical outcomes.**  
All calculations, regressions, and results are **not financial advice**.  
Past performance does **not guarantee future results**. 

Any example portfolios included in this project are **approximations of conceived factor exposure strategies**, not investment recommendations.  
Users are solely responsible for any decisions or applications derived from this toolkit.

---

## Acknowledgements

The developer thanks **Curvo.eu** and **Benjamin Felix (PWL Capital, Canada)** for permission to use selected datasets and published results.  
These have been used solely to benchmark and validate analysis code within the Jupyter notebooks.

---

## Interactive GUI (Tkinter)  
**Implemented**

![GUI](images/GUI.png)

- Simplified GUI-based portfolio builder  
- Save and import portfolios and settings  
- Toggle leveraged strategies  
- One-click access to the full analysis suite  

---

## 1. Fama–French Five-Factor (FF5) Regressions  
**Implemented**

![Factor Regression](images/Factor_Regression.png)  
![Rolling Regressions](images/Rolling_Regressions.png)

- Factor data ingestion from the Ken French Data Library  
- Portfolio- and asset-level risk factor regressions
- **Rolling regressions** over configurable windows (e.g. 3y / 5y / 10y)  
- Factor exposure, alpha, and contribution analysis  

---

## 2. Markowitz Portfolio Optimiser  
**Implemented**

![Markowitz Optimisation](images/Markowitz.png)

- Mean–variance optimisation  
- Covariance estimation from historical or factor-implied returns  
- Efficient frontier construction  
- **Ledoit–Wolf covariance matrix shrinkage**  
- Direct application of optimal weights to the active portfolio  

---

## 3. Risk Report  
**Implemented**

![Risk Report](images/Risk_Report.png)

- Historical drawdown analysis  
- Asset- and factor-level risk contribution assessment  
- Monthly CVaR and VaR at 95% confidence  

---

## 4. Correlation Matrix  
**Implemented**

![Correlation Matrix](images/Correlation_Matrix.png)

- Inter-asset correlation inspection
- Heatmap visualisation for rapid structure assessment  

---

## 5. Monte Carlo Retirement Simulation  
**Implemented**

![Monte Carlo Simulation](images/Monte_Carlo.png)

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

## Currency Support  
**Implemented**

- Automatic currency detection via Yahoo Finance metadata  
- FX normalisation to USD for cross-currency portfolios  
- Seamless integration into FF5 analysis and Monte Carlo simulations  

---

## Assumptions and Limitations

To maintain interpretability and analytical clarity, several regional, institutional, and broker-dependent real-world effects are intentionally excluded:

Key limitations include:

- All assets are converted to USD to enable cross-currency factor regressions  
- Portfolio weights are assumed to remain constant (rebalancing planned)
- Transaction costs are not modelled
- Leverage multipliers are static; real-world borrowing constraints are not modelled  
- Assets with short histories may be extended using FF5-based synthetic returns  
- Taxes are **not** modelled; withdrawals represent spending *plus* taxes  
- The Markowitz optimiser uses randomised sampling; direct solvers (e.g. `pypfopt`) would be more robust  
- Market data is sourced from Yahoo Finance; additional providers are planned  

This simplifies interpretation but may deviate from realisable outcomes.

---

## Realism Update (In Development)

Features that materially increase realism but are currently separated from the core engine:

- **Portfolio rebalancing**
- **Inflation-adjusted withdrawals**
- Limited leverage model with loan-to-value caps and margin calls  
- Real (not just nominal) return tracking  
- Stress testing with ±1% return and +20% volatility shocks  

These will be introduced as **opt-in layers**, not hardwired assumptions.

---

## Known Issues
- Currently none.
- Recent fixes:
  - Markowitz optimiser weights now correctly apply in the Treeview  
  - Negative wealth paths are terminated in Monte Carlo simulations
  - Leverage application before simulations fixed

---
Pull requests, suggestions and bug reports are welcome.

---
## Getting Started

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
