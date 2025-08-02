import pandas as pd
import numpy as np
import statsmodels.api as sm

class Asset:
    def __init__(self, name, returns):
        """
        name: str, e.g. 'AAPL'
        returns: pd.Series with datetime index, monthly returns (decimal)
        """
        self.name = name
        self.returns = returns

class Portfolio:
    def __init__(self, assets=None, weights=None):
        """
        assets: list of Asset objects
        weights: pd.DataFrame with index = dates, columns = asset names, weights as values
                 Should sum to 1 per date (for monthly rebalancing)
        """
        self.assets = assets or []
        self.weights = weights  # pd.DataFrame or None (equal weights)
    
    def add_asset(self, asset):
        self.assets.append(asset)
    
    def get_returns_df(self):
        """
        Create a DataFrame with aligned returns for each asset.
        """
        data = {}
        for asset in self.assets:
            data[asset.name] = asset.returns
        returns_df = pd.DataFrame(data)
        return returns_df.sort_index()
    
    def portfolio_returns(self):
        """
        Calculate portfolio returns over time applying the weights per date.
        If weights is None, assume equal weights.
        """
        returns_df = self.get_returns_df()
        
        if self.weights is None:
            # Equal weights
            w = np.repeat(1/len(self.assets), len(self.assets))
            port_rets = returns_df.dot(w)
        else:
            # Align weights to returns dates
            weights_aligned = self.weights.reindex(returns_df.index).fillna(method='ffill').fillna(0)
            # Multiply weights with returns row-wise and sum
            port_rets = (returns_df * weights_aligned).sum(axis=1)
        
        return port_rets
    
    def run_factor_regression(self, factor_returns_df, risk_free_rate=0.005):
        """
        factor_returns_df: DataFrame with factor returns (same dates as portfolio returns)
        risk_free_rate: float, monthly risk free rate
        
        Runs regression of excess portfolio returns on excess factors.
        """
        port_rets = self.portfolio_returns()
        df = pd.DataFrame({'Portfolio': port_rets}).join(factor_returns_df, how='inner')
        
        # Calculate excess returns
        df = df - risk_free_rate
        
        y = df['Portfolio']
        X = df.drop(columns=['Portfolio'])
        X = sm.add_constant(X)
        
        model = sm.OLS(y, X).fit()
        return model

'''
# Example usage

# Create assets
aapl = Asset('AAPL', aapl_returns_series)
spy = Asset('SPY', spy_returns_series)

# Make portfolio and add assets
p = Portfolio()
p.add_asset(aapl)
p.add_asset(spy)

# Optionally set weights (monthly rebalancing)
weights_df = pd.DataFrame({
    'AAPL': [0.6, 0.5, 0.7],
    'SPY': [0.4, 0.5, 0.3]
}, index=pd.to_datetime(['2024-01-31', '2024-02-29', '2024-03-31']))

p.weights = weights_df

# Run factor regression (using a pre-loaded factor returns DataFrame)
model = p.run_factor_regression(factor_returns_df)
print(model.summary())
'''
