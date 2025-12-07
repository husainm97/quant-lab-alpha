# withdrawal_strategies.py
from typing import Callable

def pct_of_initial_capital(initial_capital: float, rate: float, min_withdrawal: float = 0, max_withdrawal: float = float("inf")) -> Callable[[float], float]:
    """Withdraw a fixed % of initial capital, with optional guardrails."""
    def strategy(current_capital: float) -> float:
        w = initial_capital * rate
        return max(min(w, max_withdrawal), min_withdrawal)
    return strategy

def pct_of_current_capital(rate: float, min_withdrawal: float = 0, max_withdrawal: float = float("inf")) -> Callable[[float], float]:
    """Withdraw a fixed % of current capital, with optional guardrails."""
    def strategy(current_capital: float) -> float:
        w = current_capital * rate
        return max(min(w, max_withdrawal), min_withdrawal)
    return strategy

def fixed_amount(amount: float) -> Callable[[float], float]:
    """Withdraw a fixed nominal amount per year."""
    def strategy(current_capital: float) -> float:
        return min(current_capital, amount)
    return strategy

def fixed_pct_initial(initial_capital: float, rate: float = 0.04):
    """
    Fixed percentage of initial capital (e.g., 4% rule)
    """
    def withdraw(current_wealth: float) -> float:
        return initial_capital * rate
    return withdraw


def guardrail_pct(min_rate: float = 0.025, max_rate: float = 0.05, benchmark_growth: float = 1.0):
    """
    Guardrail strategy: withdraw % within min/max depending on portfolio performance
    - If portfolio above benchmark_growth, withdraw max_rate
    - If below benchmark_growth, withdraw min_rate
    benchmark_growth can be a moving average multiplier or initial capital
    """
    def withdraw(current_wealth: float) -> float:
        # Simple rule: compare to initial capital for scaling
        ratio = current_wealth / 1_000_000.0  # benchmark is initial capital
        if ratio >= 1.0:
            rate = max_rate
        elif ratio <= 0.8:
            rate = min_rate
        else:
            # linear interpolation between min and max
            rate = min_rate + (ratio - 0.8)/(1.0-0.8) * (max_rate - min_rate)

         
        return current_wealth * rate
    return withdraw

def bucket_strategy(cash_years: int = 3, annual_withdrawal: float = 50_000):
    """
    Bucket strategy: keeps a short-term cash reserve for withdrawals
    - cash_years: number of years of withdrawals to hold in cash
    - annual_withdrawal: target annual withdrawal amount
    """
    cash_buffer = annual_withdrawal * cash_years
    leftover = 0.0  # tracks how much extra cash is in buffer

    def withdraw(current_wealth: float) -> float:
        nonlocal cash_buffer, leftover

        # If portfolio is low, withdraw as much as possible but not more than current wealth
        if current_wealth < cash_buffer:
            amt = current_wealth
            leftover = 0
            cash_buffer = 0
        else:
            amt = annual_withdrawal
            leftover = current_wealth - cash_buffer
            # refill the buffer up to target
            cash_buffer = min(cash_buffer + leftover, annual_withdrawal * cash_years)

        return amt

    return withdraw
