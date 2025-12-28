import numpy as np
import os
import sys
# --- Ensure we can import local modules (portfolio.py, investor.py) ---
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import portfolio_module as portfolio_module

'''
This class is a possible future update of an investor as an 'agent' who performs actions on the portfolio to run the bootstrap sims with different "personalised" strategies.  
'''

class Investor:
    def __init__(self, name, capital, income=0.0, withdrawal_strategy=None, risk_tolerance=0.5):
        self.name = name
        self.capital = capital
        self.income = income
        self.risk_tolerance = risk_tolerance
        self.portfolio = portfolio.Portfolio(name=f"{name}'s Portfolio")
        self.withdrawal_strategy = withdrawal_strategy  # callable(current_capital) -> float

    def annual_update(self):
        """Apply income and withdraw according to strategy."""
        self.capital += self.income
        if self.withdrawal_strategy:
            withdrawal_amount = self.withdrawal_strategy(self.capital)
            self.withdraw(withdrawal_amount)

    def deposit(self, amount: float):
        """Increase capital."""
        self.capital += amount

    def withdraw(self, amount: float):
        """Withdraw capital (for consumption)."""
        if amount > self.capital:
            raise ValueError("Not enough capital.")
        self.capital -= amount

    def annual_update(self):
        """Simulate yearly income/spending."""
        self.capital += self.income
        withdrawal = self.capital * self.spending_rate
        self.withdraw(withdrawal)

    def invest(self, ticker: str, weight: float):
        """Add investment to portfolio."""
        self.portfolio.add_asset(ticker, weight)

    def rebalance(self):
        """Placeholder for strategy-based rebalancing logic."""
        self.portfolio._normalize_weights()

    def simulate_year(self, asset_returns: dict):
        """Apply portfolio return + income/spending to capital."""
        port_return = self.portfolio.expected_return(asset_returns)
        self.capital *= (1 + port_return - self.portfolio.interest_rate * (self.portfolio.leverage - 1))
        self.annual_update()

    def summary(self):
        return {
            "Investor": self.name,
            "Capital": round(self.capital, 2),
            "Income": self.income,
            "Spending Rate": self.spending_rate,
            "Risk Tolerance": self.risk_tolerance,
            "Portfolio": self.portfolio.summary(),
        }
