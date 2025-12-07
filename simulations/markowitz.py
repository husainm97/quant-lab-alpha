# markowitz.py

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def markowitz_optimize(returns, risk_free=0.0):
    """
    Markowitz mean–variance optimization.
    'returns' is a DataFrame where columns are assets and values are monthly % returns.
    Assumes no NaNs and aligned data.
    """

    # Convert to arrays
    rets = returns.mean().values              # vector of mean returns
    cov = returns.cov().values               # covariance matrix
    n = len(rets)                            # number of assets

    # Negative Sharpe (because minimize)
    def neg_sharpe(weights):
        port_ret = np.dot(weights, rets)
        port_vol = np.sqrt(weights @ cov @ weights)
        return -(port_ret - risk_free) / port_vol

    # Constraints: sum(weights) = 1
    constraints = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
    )

    # Long-only portfolio
    bounds = [(0, 1)] * n

    # Starting guess
    w0 = np.ones(n) / n

    # Solve!
    result = minimize(
        neg_sharpe, w0, method="SLSQP",
        bounds=bounds, constraints=constraints
    )

    if not result.success:
        raise RuntimeError("Markowitz optimization failed: " + result.message)

    w_opt = result.x
    port_ret = np.dot(w_opt, rets)
    port_vol = np.sqrt(w_opt @ cov @ w_opt)
    sharpe = (port_ret - risk_free) / port_vol

    return {
        "weights": w_opt,
        "expected_return": port_ret,
        "volatility": port_vol,
        "sharpe": sharpe
    }


def run_markowitz(returns_df, risk_free=0.0):
    """
    The function the GUI should import.
    returns_df: already monthly pct change returns (DataFrame)
    """
    return markowitz_optimize(returns_df, risk_free=risk_free)