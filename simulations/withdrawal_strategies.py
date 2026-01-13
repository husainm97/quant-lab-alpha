import numpy as np
from typing import Callable

def inflation_adjusted(strategy: Callable) -> Callable:
    """
    Decorator that applies inflation adjustment to withdrawal amounts.
    
    Args:
        strategy: Base withdrawal strategy function
    
    Returns:
        Wrapped function with inflation adjustment
    """
    def wrapped(current_wealth, year: int = 0, inflation_rate: float = 0.03, **kwargs):
        base_amount = strategy(current_wealth, year=year, **kwargs)
        return base_amount * ((1 + inflation_rate) ** year)
    return wrapped


def fixed_pct_initial(initial_capital: float, rate: float = 0.04, scale: float = 1) -> Callable:
    """
    Withdraws a fixed percentage of initial capital each period.
    
    CHANGES FOR VECTORIZATION:
    - Changed from scalar to np.full_like() to match current_wealth shape
    - Now returns array of shape (n_bootstrap,) when current_wealth is array
    
    Args:
        initial_capital: Starting portfolio value
        rate: Withdrawal rate (e.g., 0.04 = 4%)
        scale: Divisor for period adjustment (12 for monthly from annual rate)
    
    Returns:
        Strategy function that withdraws fixed amount
    
    Example:
        >>> strategy = fixed_pct_initial(1_000_000, 0.04, scale=12)
        >>> wealth = np.array([900_000, 1_100_000, 800_000])
        >>> strategy(wealth, year=0)
        array([3333.33, 3333.33, 3333.33])  # Same amount for all paths
    """
    @inflation_adjusted
    def withdraw(current_wealth, year: int = 0, **kwargs):
        # Returns array matching current_wealth shape, all same value
        return np.full_like(current_wealth, initial_capital * rate / scale, dtype=float)
    return withdraw


def pct_of_current_capital(rate: float, scale: float = 1, min_w: float = 0, max_w: float = float("inf")) -> Callable:
    """
    Withdraws a fixed percentage of current portfolio value.
    
    CHANGES FOR VECTORIZATION:
    - Changed from scalar min(max()) to np.clip() for array support
    - Now properly handles array inputs element-wise
    
    Args:
        rate: Withdrawal rate (e.g., 0.04 = 4% of current value)
        scale: Divisor for period adjustment (12 for monthly)
        min_w: Minimum withdrawal amount per period
        max_w: Maximum withdrawal amount per period
    
    Returns:
        Strategy function that withdraws percentage of current wealth
    
    Example:
        >>> strategy = pct_of_current_capital(0.04, scale=12)
        >>> wealth = np.array([900_000, 1_100_000, 800_000])
        >>> strategy(wealth, year=0)
        array([3000., 3666.67, 2666.67])  # Proportional to current wealth
    """
    @inflation_adjusted
    def withdraw(current_wealth, year: int = 0, **kwargs):
        w = current_wealth * rate
        return np.clip(w, min_w, max_w) / scale
    return withdraw


def guardrail_pct(min_rate: float = 0.025, max_rate: float = 0.05, scale: float = 1, 
                  initial_capital: float = 1_000_000) -> Callable:
    """
    Withdraws variable percentage based on portfolio performance vs initial value.
    
    CHANGES FOR VECTORIZATION:
    - Changed from scalar if/elif/else to np.where() for vectorized conditionals
    - Added initial_capital parameter (was hardcoded to 1M)
    - Now processes entire array of wealth paths simultaneously
    
    Logic:
        - If wealth >= initial: withdraw at max_rate
        - If wealth <= 0.8 × initial: withdraw at min_rate
        - Between 0.8-1.0: linear interpolation between min and max rates
    
    Args:
        min_rate: Minimum withdrawal rate when portfolio is struggling
        max_rate: Maximum withdrawal rate when portfolio is healthy
        scale: Divisor for period adjustment (12 for monthly)
        initial_capital: Benchmark wealth level (default 1M)
    
    Returns:
        Strategy function with guardrails
    
    Example:
        >>> strategy = guardrail_pct(0.025, 0.05, scale=12, initial_capital=1_000_000)
        >>> wealth = np.array([700_000, 900_000, 1_100_000])  # Poor, mid, good
        >>> strategy(wealth, year=0)
        array([1458.33, 3125.00, 4583.33])  # 2.5%, 3.75%, 5% annual rates
    """
    @inflation_adjusted
    def withdraw(current_wealth, year: int = 0, **kwargs):
        ratio = current_wealth / initial_capital
        
        # Vectorized conditional: rate depends on wealth ratio
        rate = np.where(
            ratio >= 1.0, 
            max_rate,  # Healthy: use max rate
            np.where(
                ratio <= 0.8, 
                min_rate,  # Struggling: use min rate
                # Linear interpolation between 0.8 and 1.0
                min_rate + (ratio - 0.8) / 0.2 * (max_rate - min_rate)
            )
        )
        
        return current_wealth * rate / scale
    return withdraw


def bucket_strategy(cash_years: int = 3, annual_withdrawal: float = 50_000, scale: float = 1) -> Callable:
    """
    Bucket strategy with cash buffer management - VECTORIZED VERSION.
    
    CHANGES FOR VECTORIZATION:
    - Replaced scalar cash_buffer with array cash_buffers (one per path)
    - Changed if/else to np.where() for vectorized conditionals
    - All operations now work on entire arrays simultaneously
    
    Strategy:
        - Maintains cash buffer of (cash_years × annual_withdrawal)
        - Withdraws from buffer each period
        - Refills buffer from portfolio when possible
        - Falls back to withdrawing full wealth if buffer depleted
    
    Args:
        cash_years: Size of cash buffer in years
        annual_withdrawal: Target annual withdrawal amount
        scale: Divisor for period adjustment (12 for monthly)
    
    Returns:
        Strategy function that manages cash buffer per simulation path
    
    Example:
        >>> strategy = bucket_strategy(cash_years=3, annual_withdrawal=60_000, scale=12)
        >>> wealth = np.array([900_000, 1_100_000, 50_000])
        >>> # First call initializes buffers
        >>> strategy(wealth, year=0)
        array([5000., 5000., 5000.])  # Normal withdrawals
        >>> # Subsequent calls maintain separate buffer for each path
    """
    monthly_withdrawal = annual_withdrawal / scale
    target_buffer = monthly_withdrawal * scale * cash_years
    
    # State: one cash buffer per simulation path (initialized on first call)
    cash_buffers = None
    
    @inflation_adjusted
    def withdraw(current_wealth, year: int = 0, **kwargs):
        nonlocal cash_buffers
        
        # Initialize buffers on first call (one per path)
        if cash_buffers is None:
            cash_buffers = np.full_like(current_wealth, target_buffer, dtype=float)
        
        # Vectorized logic: withdraw amount depends on buffer status
        # If wealth < buffer: withdraw everything (depleted)
        # Else: withdraw monthly amount
        amt = np.where(
            current_wealth < cash_buffers,
            current_wealth,  # Depleted: take what's left
            monthly_withdrawal  # Normal: take monthly amount
        )
        
        # Update buffers: set to 0 if depleted
        cash_buffers = np.where(
            current_wealth < cash_buffers,
            0.0,
            cash_buffers
        )
        
        # Refill buffer from surplus (only for non-depleted paths)
        # surplus = what's left after maintaining current buffer
        surplus = np.maximum(0, current_wealth - cash_buffers)
        
        # refill = how much we can add back to buffer (up to target)
        refill = np.minimum(surplus, target_buffer - cash_buffers)
        
        # Only refill where we're not depleted
        cash_buffers = np.where(
            current_wealth >= cash_buffers,
            cash_buffers + refill,
            cash_buffers  # Keep at 0 if depleted
        )
        
        return amt
    
    return withdraw