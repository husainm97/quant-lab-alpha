# quant-lab-alpha - Portfolio Factor Regression & Analysis Toolkit

**Quant Lab** is a Python toolkit for portfolio analysis using the Fama-French 5-factor model and related quantitative finance tools. It helps you:

- Download and process Fama-French factor data  
- Calculate factor premiums over user-defined time periods (e.g., for academic paper replication or custom analysis)  
- Build regressors to analyze portfolio returns against factor premiums  
- Evaluate portfolio performance and risk metrics  
- Easily extend the toolkit with additional quantitative finance methods (volatility tests, Markowitz optimization, etc.)

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
