import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Example functions, build in progress

def compute_sharpe_ratio(returns, risk_free=0.0):
    excess_returns = returns - risk_free
    return excess_returns.mean() / excess_returns.std()

def compute_cagr(returns):
    """
    Compound Annual Growth Rate from monthly log returns.
    """
    n_years = (returns.index[-1] - returns.index[0]).days / 365.25
    cumulative_return = np.exp(returns.sum())
    return cumulative_return**(1/n_years) - 1

def compute_max_drawdown(cumulative_returns):
    """
    Takes cumulative returns (not log) and finds max drawdown.
    """
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()

def run_ff_regression(portfolio_returns, ff_factors):
    """
    Runs a 5-factor regression and returns the fitted model.
    """
    excess_returns = portfolio_returns - ff_factors['RF']
    X = ff_factors.drop(columns="RF")
    X = sm.add_constant(X)
    model = sm.OLS(excess_returns, X).fit()
    return model

def plot_cumulative_returns(returns, title="Portfolio"):
    cumulative = np.exp(returns.cumsum())
    plt.figure(figsize=(10, 5))
    plt.plot(cumulative, label="Cumulative Return")
    plt.title(f"{title} – Cumulative Return")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def summary_report(returns, ff_factors):
    """
    Prints a summary of performance + regression stats.
    """
    sharpe = compute_sharpe_ratio(returns, ff_factors['RF'])
    cagr = compute_cagr(returns)
    cumulative = np.exp(returns.cumsum())
    max_dd = compute_max_drawdown(cumulative)

    print(f"   Portfolio Summary:")
    print(f" - Sharpe Ratio:       {sharpe:.2f}")
    print(f" - CAGR:               {cagr:.2%}")
    print(f" - Max Drawdown:       {max_dd:.2%}\n")

    model = run_ff_regression(returns, ff_factors)
    print("Fama-French 5-Factor Regression:\n")
    print(model.summary())

    return model
