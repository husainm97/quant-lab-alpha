# -----------------------------
# File: examples/runner.py
# -----------------------------
"""Example usage of the portfolio_bootstrap package.


Save file and run `python examples/runner.py` after installing package folder next to it.
"""
import numpy as np
import pandas as pd
from portfolio_bootstrap import monte_carlo_simulation, FixedPercentWithdrawal, GuardrailsWithdrawal




def main():
# toy historical monthly returns (e.g. 12 months repeated) - in decimals
hist = np.array([0.01, 0.02, -0.015, 0.03, -0.02, 0.005, 0.01, 0.015, -0.01, 0.02, 0.025, 0.0])


n_months = 30 * 12 # 30 years monthly
sims = 500


# example 4% fixed-percent (based on initial capital)
w = FixedPercentWithdrawal(annual_rate=0.04, initial_capital=1_000_000)


df = monte_carlo_simulation(
returns=hist,
n_periods=n_months,
n_sims=sims,
block_len=12,
initial_capital=1_000_000,
leverage=1.0,
financing_rate_annual=0.0,
withdraw_strategy=w,
random_seed=42,
progress=True
)


# compute percentile outcomes
terminal = df.iloc[-1, :]
summary = pd.Series({
'median': np.percentile(terminal, 50),
'p10': np.percentile(terminal, 10),
'p90': np.percentile(terminal, 90),
})
print(summary)




if __name__ == '__main__':
main()
