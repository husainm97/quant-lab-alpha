# quant-lab-alpha - Portfolio Factor Regression & Analysis Toolkit

Quant Lab is a Python toolkit for portfolio analysis built around the Fama–French 5-Factor Model and related quantitative finance methods. It provides a streamlined workflow to:

- Download and preprocess Fama–French factor datasets
- Compute factor premiums over custom time horizons (for academic replication or bespoke analysis)
- Run regressions of portfolio returns against factor exposures
- Evaluate performance and risk metrics for portfolios
- Extend with additional methods (e.g. volatility tests, Markowitz optimization, custom factors)
- Analyze UCITS ETFs and construct optimal portfolios under the Five-Factor framework
- Explore explanatory Jupyter notebooks with clear, reproducible examples

---

## Features

- Data fetcher from official sources (Ken French data library, yahoo finance, etc)  
- Compute factor premiums for custom date ranges  
- Perform multi-factor regressions on user portfolios  
- Support for multiple portfolio input formats (e.g., CSV, Yahoo Finance tickers)  
- Modular design for easy expansion with new tools and models  

---

## Getting Started

1. Clone the repo:  
   ```bash
   git clone https://github.com/yourusername/quant-lab-alpha.git
   cd quant-lab-alpha

2. Install Dependencies:  
   ```bash
   pip install -r requirements.txt
