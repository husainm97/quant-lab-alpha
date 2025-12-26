# quant-lab-alpha — Portfolio Factor Regression & Analysis Toolkit

**Quant Lab Alpha** is a Python toolkit for systematic portfolio analysis and factor-based investing.  
It is built around the **Fama–French Five-Factor (FF5) Model**, with extensions into Monte Carlo simulation and classical portfolio optimization.

The framework provides a clean, reproducible workflow for **academic replication**, **portfolio construction**, and **long-horizon outcome analysis**.

---

## Disclaimer

**This repository is strictly for educational and research purposes.**  
All calculations, regressions, and results are **not financial advice**.  
Past performance does **not guarantee future results**.  

Any example portfolios included in this project are **approximations of conceived factor exposure strategies**, not investment recommendations.  
Users are solely responsible for any decisions or applications derived from this toolkit.

---

## Acknowledgements

The developer thanks **Curvo.eu** and **Benjamin Felix (PWL Capital, Canada)** for permission to use selected datasets and published results.  
These have been used solely to benchmark and validate analysis code within the Jupyter notebooks.

---

## Current Project Scope (Status: Active)

### 1. Fama–French Five-Factor (FF5) Analysis  
**Implemented**

- Factor data ingestion from the Ken French Data Library  
- Portfolio- and asset-level FF5 regressions  
- **Rolling regressions** over configurable windows (e.g. 3y / 5y / 10y)  
- Factor exposure, alpha, and contribution analysis  

---

### 2. Monte Carlo Portfolio Simulation  
**Implemented**

- Synthetic return histories generated via FF5-fitted models  
- Block / bootstrap-style simulations  
- Static-weight portfolios  
- Multiple withdrawal strategies:
  - Fixed 4% (initial capital)
  - Variable 4% of current capital
  - Guardrails (2.5–5%)
  - Bucket strategy
- Failure probability and terminal wealth analysis  

**In Development**

- Sharpe ratio, drawdown, and richer risk diagnostics  
- Improved stress testing and scenario labeling  

---

### 3. Markowitz Portfolio Optimizer  
**Implemented**

- Mean–variance optimization  
- Covariance estimation from historical or factor-implied returns  
- Efficient frontier construction  
- Optimal portfolios under classical Markowitz assumptions  

---

## Currency Support

**Implemented**

- Automatic currency detection via Yahoo Finance metadata  
- FX normalization to USD for cross-currency portfolios  
- Seamless integration into FF5 analysis and Monte Carlo simulations  

---

## Realism Update™ (In Development)

Features that materially increase realism but are intentionally separated to keep the core engine clean and interpretable:

- **Portfolio rebalancing**
- **Inflation-adjusted withdrawals**
- Real (not just nominal) return tracking  
- Stress testing with ±1% return and +20% volatility shocks  

These features are planned as **opt-in layers**, not hardwired assumptions.

---

## Assumptions and Limitations

- Portfolio weights are currently assumed constant; drift and rebalancing are planned in a future update  
- Assets with short histories are extended using synthetic returns generated from FF5 regressions  
- Taxes are **not** modeled; withdrawals currently represent spending *plus* taxes  
- The Markowitz optimizer runs on a limited random sample; direct solvers (e.g. `pypfopt`) would yield more robust solutions  
- Primary market data is sourced from Yahoo Finance; additional data providers are planned  

---

## Known Issues

- Negative-wealth Monte Carlo paths are currently visualized only for the fixed 4% strategy  
- Interest on negative wealth is not yet modeled  
- Markowitz weights applied via the GUI update the Portfolio object but do not yet refresh the Treeview  

---

## Getting Started

### 1. Clone the repository
```
bash
git clone https://github.com/husainm97/quant-lab-alpha.git
cd quant-lab-alpha
```

### 1. **Launch the GUI**
```
python gui.py
```
