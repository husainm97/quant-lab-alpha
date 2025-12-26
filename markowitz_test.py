import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# ------------------------
# 1. Define assets & download data
# ------------------------
tickers = [
    "SPY",  # US equity
    "EFA",  # Developed world equity
    "AGG",  # US bonds
    "BNDX", # Global bonds
    "MTUM", # Momentum factor ETF
    "VLUE", # Value factor ETF
]

start_date = "1900-01-01"
end_date = datetime.today().strftime('%Y-%m-%d')

# Use Adj Close consistently
prices = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)["Adj Close"]

# ------------------------
# 2. Convert daily prices to monthly returns
# ------------------------
monthly_prices = prices.resample('ME').last()
monthly_returns_all = monthly_prices.pct_change()

# ------------------------
# 2b. Align to maximal common range (Tkinter style)
# ------------------------
series_list = [monthly_returns_all[t] for t in tickers]

start_common = max(s.index.min() for s in series_list)
end_common = min(s.index.max() for s in series_list)

monthly_returns = pd.concat([s.loc[start_common:end_common] for s in series_list], axis=1).dropna(how='any')
monthly_returns.columns = tickers  # enforce original tickers order

print(f"Using common range: {monthly_returns.index.min().date()} to {monthly_returns.index.max().date()}")
print(f"Number of months: {len(monthly_returns)}")

# ------------------------
# 3. Monte Carlo Markowitz frontier
# ------------------------
def monte_carlo_frontier(returns: pd.DataFrame, risk_free_annual=0.02, n_portfolios=50000):
    rf_monthly = (1 + risk_free_annual)**(1/12) - 1
    mean_rets = returns.mean().values
    cov = returns.cov().values
    n_assets = len(mean_rets)

    # generate random long-only weights
    weights = np.random.rand(n_portfolios, n_assets)
    weights /= weights.sum(axis=1)[:, None]

    port_rets = weights @ mean_rets
    port_vols = np.sqrt(np.einsum('ij,jk,ik->i', weights, cov, weights))
    sharpe = (port_rets - rf_monthly) / port_vols

    # tangent portfolio
    idx_tangent = np.argmax(sharpe)
    tangent_w = weights[idx_tangent]
    tangent_ret = port_rets[idx_tangent]
    tangent_vol = port_vols[idx_tangent]

    return port_rets, port_vols, sharpe, tangent_w, tangent_ret, tangent_vol

port_rets, port_vols, sharpe, tangent_w, tangent_ret, tangent_vol = monte_carlo_frontier(monthly_returns)

# ------------------------
# 4. Plot efficient frontier + tangent
# ------------------------
rf_monthly = (1 + 0.02)**(1/12) - 1
plt.figure(figsize=(10,6))
plt.scatter(port_vols, port_rets, c=sharpe, cmap='viridis', s=10, alpha=0.5)
plt.colorbar(label='Sharpe Ratio')

# Tangent portfolio
plt.scatter(tangent_vol, tangent_ret, color='r', s=100, label='Tangent Portfolio')

# Capital Market Line
x_line = np.linspace(0, tangent_vol*1.5, 50)
slope = (tangent_ret - rf_monthly) / tangent_vol
y_line = rf_monthly + slope * x_line
plt.plot(x_line, y_line, 'r--', label='Capital Market Line')

# Annotate tangent weights
weights_text = "\n".join(f"{t}: {w:.2f}" for t, w in zip(monthly_returns.columns, tangent_w))
plt.annotate(weights_text,
             xy=(tangent_vol, tangent_ret),
             xytext=(tangent_vol*1.1, tangent_ret*0.95),
             arrowprops=dict(facecolor='black', arrowstyle="->"),
             fontsize=10, bbox=dict(boxstyle='round,pad=0.3', alpha=0.2))

plt.xlabel('Volatility (σ)')
plt.ylabel('Expected Return')
plt.title('Monte Carlo Markowitz Frontier (Tkinter-aligned)')
plt.grid(True)
plt.legend()
plt.show()

# ------------------------
# 5. Print suggested weights
# ------------------------
print("=== Tangent Portfolio Weights ===")
for t, w in zip(monthly_returns.columns, tangent_w):
    print(f"{t}: {w:.4f}")
