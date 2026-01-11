# withdrawal_strategies.py
from typing import Callable

# --- Decorator ---
def inflation_adjusted(strategy: Callable[[float, int], float]) -> Callable[[float, int, float], float]:
    """
    Decorator to adjust a withdrawal strategy for inflation.
    Expects the decorated function to take (current_wealth, year=0, **kwargs).
    The inflation_rate is now passed dynamically at call time.
    """
    def wrapped(current_wealth: float, year: int = 0, inflation_rate: float = 0.03, **kwargs) -> float:
        base_amount = strategy(current_wealth, year=year, **kwargs)
        # Apply compounded inflation
        return base_amount * ((1 + inflation_rate) ** year)
    return wrapped


def pct_of_current_capital(rate: float, scale: float = 1, min_withdrawal: float = 0, max_withdrawal: float = float("inf")) -> Callable[[float, int, float], float]:
    """Withdraw a fixed % of current capital, with optional guardrails."""
    @inflation_adjusted
    def withdraw(current_wealth: float, year: int = 0, **kwargs) -> float:
        w = current_wealth * rate
        return max(min(w, max_withdrawal), min_withdrawal) / scale
    return withdraw


def fixed_pct_initial(initial_capital: float, rate: float = 0.04, scale: float = 1) -> Callable[[float, int, float], float]:
    """Fixed percentage of initial capital (e.g., 4% rule)."""
    @inflation_adjusted
    def withdraw(current_wealth: float, year: int = 0, **kwargs) -> float:
        return initial_capital * rate / scale
    return withdraw


def guardrail_pct(min_rate: float = 0.025, max_rate: float = 0.05, scale: float = 1) -> Callable[[float, int, float], float]:
    """
    Guardrail strategy: withdraw % within min/max depending on portfolio performance.
    """
    @inflation_adjusted
    def withdraw(current_wealth: float, year: int = 0, **kwargs) -> float:
        ratio = current_wealth / 1_000_000.0  # benchmark is initial capital
        if ratio >= 1.0:
            rate = max_rate
        elif ratio <= 0.8:
            rate = min_rate
        else:
            rate = min_rate + (ratio - 0.8) / (1.0 - 0.8) * (max_rate - min_rate)
        return current_wealth * rate / scale
    return withdraw


def bucket_strategy(cash_years: int = 3, annual_withdrawal: float = 50_000, scale: float = 1) -> Callable[[float, int, float], float]:
    """
    Bucket strategy with consistent monthly units.
    """
    monthly_withdrawal = annual_withdrawal / scale
    target_buffer = monthly_withdrawal * scale * cash_years
    cash_buffer = target_buffer

    @inflation_adjusted
    def withdraw(current_wealth: float, year: int = 0, **kwargs) -> float:
        nonlocal cash_buffer

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
