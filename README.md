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

![GUI](images/GUI.png)

- Simplified GUI-based portfolio builder  
- Save and import portfolios and settings  
- Toggle leveraged strategies  
- One-click access to the full analysis suite  

---

## 1️⃣ Fama–French Five-Factor (FF5) Regressions  
**Implemented**

![Factor Regression](images/Factor_Regression.png)  
![Rolling Regressions](images/Rolling_Regressions.png)

- Factor data ingestion from the Ken French Data Library  
- Portfolio- and asset-level risk factor regressions  
- **Rolling regressions** over configurable windows (e.g. 3y / 5y / 10y)  
- Factor exposure, alpha, and contribution analysis  

---

## 2️⃣ Markowitz Portfolio Optimiser  
**Implemented**

![Markowitz Optimisation](images/Markowitz.png)

- Mean–variance optimisation  
- Covariance estimation from historical or factor-implied returns  
- Efficient frontier construction  
- **Ledoit–Wolf covariance matrix shrinkage**  
- Direct application of optimal weights to the active portfolio  

---

## 3️⃣ Risk Report  
**Implemented**

![Risk Report](images/Risk_Report.png)

- Historical drawdown analysis  
- Asset- and factor-level risk contribution assessment  
- Monthly CVaR and VaR at 95% confidence  

---

## 4️⃣ Correlation Matrix  
**Implemented**

![Correlation Matrix](images/Correlation_Matrix.png)

- Inter-asset correlation inspection  
- Heatmap visualisation for rapid structure assessment  

---

## 5️⃣ Monte Carlo Retirement Simulation  
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

## 💱 Currency Support  
**Implemented**

- Automatic currency detection via Yahoo Finance metadata  
- FX normalisation to USD for cross-currency portfolios  
- Seamless integration into FF5 analysis and Monte Carlo simulations  

---

## ⚖️ Assumptions and Limitations

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

## 🧪 Realism Update (In Development)

Features that materially increase realism but are currently separated from the core engine:

- **Portfolio rebalancing**  
- **Inflation-adjusted withdrawals**  
- Limited leverage model with loan-to-value caps and margin calls  
- Real (not just nominal) return tracking  

These will be introduced as **opt-in layers**, not hardwired assumptions.

---

## 🐞 Known Issues

- Currently none. Please open an issue or contact the developer to report a bug.  
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
## 🐳 Docker Setup

This project includes Docker configuration for a reproducible development environment with all dependencies pre-installed.

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop) installed and running

### Quick Start with Docker

1. **Clone the repository:**
```bash
   git clone https://github.com/dhruva-divate/quant-lab-alpha.git
   cd quant-lab-alpha
```

2. **Build the Docker image:**
```bash
   docker-compose build
```

3. **Start the container:**
```bash
   docker-compose up
```

4. **Access Jupyter Lab:**
   Open your browser and navigate to:
```
   http://localhost:8888
```

### Daily Workflow
```bash
# Start working
docker-compose up

# Access Jupyter Lab at http://localhost:8888
# Make your changes in notebooks or code

# Stop the container (Ctrl+C in terminal, or:)
docker-compose down
```

### Running the GUI

**Note:** The Tkinter GUI (`main.py`) requires a display and is best run locally on your machine:
```bash
# Install dependencies locally
pip install -r requirements.txt

# Run the GUI
python main.py
```

For Docker users who want to run the GUI, X11 forwarding setup is required (advanced).

### Docker Commands
```bash
# Start container in background (detached mode)
docker-compose up -d

# View container logs
docker-compose logs

# Stop container
docker-compose down

# Rebuild after changes to Dockerfile or requirements.txt
docker-compose build --no-cache

# Open a shell inside the container
docker exec -it quant-lab-alpha bash

# Run Python scripts inside container
docker exec -it quant-lab-alpha python your_script.py
```

### What's Included

The Docker environment provides:
- **Python 3.11** with all project dependencies
- **Jupyter Lab** for interactive development
- **Persistent volumes** for Jupyter settings
- **Live code editing** - changes to local files are immediately reflected in the container

### Advantages of Docker

 **Reproducible environment** - Same setup on any machine  
 **No dependency conflicts** - Isolated from your system Python  
 **Easy collaboration** - Share exact development environment  
 **Quick onboarding** - New contributors just run `docker-compose up`

---
