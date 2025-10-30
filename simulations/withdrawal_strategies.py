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

# You can add more strategies here in the future
