# withdrawal_strategies.py
from typing import Callable

def pct_of_current_capital(rate: float, scale: float = 1, min_withdrawal: float = 0, max_withdrawal: float = float("inf")) -> Callable[[float], float]:
    """Withdraw a fixed % of current capital, with optional guardrails."""
    def strategy(current_capital: float) -> float:
        w = current_capital * rate
        return max(min(w, max_withdrawal), min_withdrawal) / scale 
    return strategy

def fixed_pct_initial(initial_capital: float, rate: float = 0.04, scale: float = 1):
    """
    Fixed percentage of initial capital (e.g., 4% rule)
    """
    def withdraw(current_wealth: float) -> float:
        return initial_capital * rate / scale 
    return withdraw


def guardrail_pct(min_rate: float = 0.025, max_rate: float = 0.05, scale: float = 1):
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

         
        return current_wealth * rate / scale 
    return withdraw

def bucket_strategy(
    cash_years: int = 3,
    annual_withdrawal: float = 50_000,
    scale: float = 1,
):
    """
    Bucket strategy with consistent monthly units.

    - cash_years: years of withdrawals to hold in cash
    - annual_withdrawal: annual target withdrawal
    - scale: scale per withdrawal period (12 = monthly, 1 = annual)
    """

    monthly_withdrawal = annual_withdrawal / scale
    target_buffer = monthly_withdrawal * scale * cash_years
    cash_buffer = target_buffer

    def withdraw(current_wealth: float) -> float:
        nonlocal cash_buffer

        # If portfolio cannot sustain the buffer, liquidate
        if current_wealth < cash_buffer:
            amt = current_wealth
            cash_buffer = 0.0
        else:
            amt = monthly_withdrawal

            # Refill buffer if possible
            surplus = current_wealth - cash_buffer
            refill = min(surplus, target_buffer - cash_buffer)
            cash_buffer += refill

        return amt

    return withdraw