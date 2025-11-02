"""
Baseline strategies for comparison
1. DJIA Buy-and-Hold
2. Min-Variance Portfolio
"""
import numpy as np
import pandas as pd
from typing import Dict, List
from config import INITIAL_BALANCE, DOW30_TICKERS
from scipy.optimize import minimize

def djia_buy_and_hold(df: pd.DataFrame, 
                     tickers: List[str] = None,
                     initial_balance: float = INITIAL_BALANCE) -> Dict:
    """
    Simple buy-and-hold strategy: Buy equal-weighted portfolio on first day
    
    Args:
        df: DataFrame with stock prices
        tickers: List of stock tickers
        initial_balance: Starting capital
    
    Returns:
        Portfolio history dictionary
    """
    tickers = tickers or DOW30_TICKERS
    
    # Get first day prices
    first_date = df.index[0]
    dates = df.index
    n_stocks = len(tickers)
    
    # Calculate prices for each stock
    prices = {}
    for ticker in tickers:
        close_col = f"{ticker}_Close"
        if close_col in df.columns:
            prices[ticker] = df[close_col].values
        else:
            prices[ticker] = np.zeros(len(dates))
    
    # Equal-weight portfolio
    weight_per_stock = 1.0 / n_stocks
    investment_per_stock = initial_balance * weight_per_stock
    
    # Buy on first day
    shares_held = {}
    for ticker in tickers:
        first_price = prices[ticker][0]
        if first_price > 0:
            shares_held[ticker] = investment_per_stock / first_price
        else:
            shares_held[ticker] = 0
    
    # Calculate portfolio value over time
    portfolio_values = []
    for i in range(len(dates)):
        total_value = 0
        for ticker in tickers:
            total_value += shares_held[ticker] * prices[ticker][i]
        portfolio_values.append(total_value)
    
    return {
        'dates': dates,
        'values': portfolio_values,
        'actions': [[0] * n_stocks] * len(dates),  # No trading
        'rewards': np.diff([initial_balance] + portfolio_values).tolist()
    }

def min_variance_portfolio(df: pd.DataFrame,
                          tickers: List[str] = None,
                          initial_balance: float = INITIAL_BALANCE,
                          rebalance_freq: int = 20) -> Dict:
    """
    Minimum variance portfolio using Markowitz optimization
    Rebalances every N days
    
    Args:
        df: DataFrame with stock prices
        tickers: List of stock tickers
        initial_balance: Starting capital
        rebalance_freq: Rebalance every N days
    
    Returns:
        Portfolio history dictionary
    """
    tickers = tickers or DOW30_TICKERS
    
    dates = df.index
    
    # Get prices
    prices = {}
    valid_tickers = []
    for ticker in tickers:
        close_col = f"{ticker}_Close"
        if close_col in df.columns:
            prices[ticker] = df[close_col].values
            valid_tickers.append(ticker)
    
    if len(valid_tickers) == 0:
        # Fallback to buy-and-hold if no data
        return djia_buy_and_hold(df, tickers, initial_balance)
    
    # Calculate returns
    returns_data = []
    for ticker in valid_tickers:
        returns = np.diff(prices[ticker]) / prices[ticker][:-1]
        returns_data.append(returns)
    
    returns_df = pd.DataFrame(np.array(returns_data).T, columns=valid_tickers)
    returns_df.index = dates[1:]
    
    # Portfolio tracking
    portfolio_values = [initial_balance]
    shares_held = {ticker: 0 for ticker in valid_tickers}
    current_balance = initial_balance
    
    actions_history = []
    
    def portfolio_variance(weights):
        """Calculate portfolio variance"""
        portfolio_return = np.sum(returns_window.mean() * weights)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return portfolio_std
    
    # Optimize and rebalance
    for i in range(len(dates)):
        if i == 0:
            # Initial buy with equal weights
            weight_per_stock = 1.0 / len(valid_tickers)
            investment_per_stock = initial_balance * weight_per_stock
            
            for ticker in valid_tickers:
                price = prices[ticker][i]
                if price > 0:
                    shares_held[ticker] = investment_per_stock / price
                else:
                    shares_held[ticker] = 0
            
            portfolio_values.append(initial_balance)
            actions_history.append([weight_per_stock] * len(tickers))
            continue
        
        # Calculate current portfolio value
        portfolio_value = current_balance
        for ticker in valid_tickers:
            portfolio_value += shares_held[ticker] * prices[ticker][i]
        portfolio_values.append(portfolio_value)
        
        # Rebalance if needed
        if i % rebalance_freq == 0 and i > rebalance_freq:
            # Use lookback window for optimization
            lookback = min(252, i)
            returns_window = returns_df.iloc[max(0, i-lookback):i]
            
            if len(returns_window) > len(valid_tickers):
                # Calculate covariance matrix
                cov_matrix = returns_window.cov().values
                
                # Optimization: minimize variance
                n = len(valid_tickers)
                constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})  # Weights sum to 1
                bounds = [(0, 1) for _ in range(n)]  # Long-only
                initial_weights = np.array([1/n] * n)
                
                try:
                    result = minimize(
                        portfolio_variance,
                        initial_weights,
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints
                    )
                    
                    if result.success:
                        optimal_weights = result.x
                        
                        # Rebalance to new weights
                        for j, ticker in enumerate(valid_tickers):
                            target_value = portfolio_value * optimal_weights[j]
                            current_value = shares_held[ticker] * prices[ticker][i]
                            
                            if target_value > current_value:
                                # Buy more
                                cost = target_value - current_value
                                if cost <= current_balance:
                                    shares_to_buy = cost / prices[ticker][i]
                                    shares_held[ticker] += shares_to_buy
                                    current_balance -= cost
                            else:
                                # Sell
                                value_to_sell = current_value - target_value
                                shares_to_sell = value_to_sell / prices[ticker][i]
                                shares_held[ticker] -= shares_to_sell
                                current_balance += value_to_sell
                        
                        # Convert to action format
                        action = [0] * len(tickers)
                        for j, ticker in enumerate(valid_tickers):
                            ticker_idx = tickers.index(ticker)
                            if ticker_idx >= 0:
                                action[ticker_idx] = optimal_weights[j] - (1/len(tickers))
                        actions_history.append(action)
                    else:
                        actions_history.append([0] * len(tickers))
                except:
                    actions_history.append([0] * len(tickers))
            else:
                actions_history.append([0] * len(tickers))
        else:
            actions_history.append([0] * len(tickers))
    
    # Calculate rewards
    rewards = np.diff(portfolio_values).tolist()
    
    return {
        'dates': dates,
        'values': portfolio_values[1:],  # Skip initial
        'actions': actions_history,
        'rewards': rewards
    }

