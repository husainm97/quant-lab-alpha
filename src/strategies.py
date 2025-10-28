# -----------------------------
self.annual_rate = annual_rate
self.freq = frequency_months
self.initial_capital = initial_capital
self._init_set = initial_capital is not None


def withdraw(self, portfolio: float, month_index: int) -> float:
    if (not self._init_set) and portfolio is not None:
        # set initial from the first call
        if self.initial_capital is None:
            self.initial_capital = portfolio
            self._init_set = True
            per_period = (self.annual_rate * (self.initial_capital if self.initial_capital is not None else portfolio)) / 12.0
    return per_period


class GuardrailsWithdrawal(WithdrawalStrategy):
    def __init__(self, base_annual_rate: float, floor_pct: float = 0.8, ceil_pct: float = 1.2, step_reduce: float = 0.1):
        """A simple guardrails mechanic inspired by Guyton-Klinger.


        - base_annual_rate: initial withdrawal rate (e.g. 0.04).
        - floor_pct/ceil_pct: relative portfolio thresholds (fractions of initial portfolio) to reduce or raise withdrawals.
        - step_reduce: multiplicative step when cutting withdrawals (e.g. 0.1 => reduce by 10%).
        """
        self.base = base_annual_rate
        self.floor = floor_pct
        self.ceil = ceil_pct
        self.step = step_reduce
        self.initial_portfolio = None
        self.current_rate = base_annual_rate


    def withdraw(self, portfolio: float, month_index: int) -> float:
        if self.initial_portfolio is None:
            self.initial_portfolio = portfolio
        # check thresholds annually at month_index == 0 mod 12
        if month_index % 12 == 0 and month_index > 0:
            ratio = portfolio / self.initial_portfolio if self.initial_portfolio > 0 else 0.0
        if ratio < self.floor:
            # cut the withdrawal rate
            self.current_rate *= (1.0 - self.step)
        elif ratio > self.ceil:
            # raise withdrawal rate a bit, but not above some cap
            self.current_rate *= (1.0 + self.step)
        # return monthly withdrawal based on current_rate and initial portfolio
        return (self.current_rate * self.initial_portfolio) / 12.0





def apply_leverage(returns: Sequence[float], leverage: float, financing_rate_annual: float = 0.0):
    """Apply leverage to a sequence of periodic returns and return net leveraged returns.

    Formula used: net_ret = L*r - (L-1)*fin_rate_periodic

    Args:
    returns: sequence of periodic returns (decimals).
    leverage: leverage factor L.
    financing_rate_annual: annual financing/carry rate.

    Returns:
    numpy array of net periodic returns after leverage cost.
    """
    arr = np.asarray(returns, dtype=float)
    fr = (1.0 + financing_rate_annual) ** (1.0 / 12.0) - 1.0
    net = leverage * arr - max(0.0, leverage - 1.0) * fr
    return net
