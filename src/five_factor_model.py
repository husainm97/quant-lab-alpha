import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Example functions, build in progress

def simulate_factor_data(num_months=1200, seed=42):
    np.random.seed(seed)
    dates = pd.date_range(start='1926-01-01', periods=num_months, freq='M')
    
    market = np.random.normal(0.071, 0.005, num_months)
    size = np.random.normal(0.08, 0.04, num_months)
    value = np.random.normal(0.095, 0.04, num_months)
    
    etf = 0.84 * market + 0.1 * size + 0.06 * value + np.random.normal(0, 0.015, num_months)
    
    df = pd.DataFrame({
        'Date': dates,
        'ETF': etf,
        'Market': market,
        'Size': size,
        'Value': value
    }).set_index('Date')
    
    return df


def compute_excess_returns(df, rf=0.005):
    return df - rf


def run_factor_regression(returns_df, target_col, factor_cols, add_const=True):
    y = returns_df[target_col]
    X = returns_df[factor_cols]
    if add_const:
        X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model


def plot_factor_histogram(returns_df, column='ETF'):
    plt.hist(returns_df[column], bins=50)
    plt.title(f"Histogram of {column} Returns")
    plt.grid(True)
    plt.show()


def factor_summary_report(model):
    print(model.summary())
