"""
Evaluation metrics and performance analysis
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from config import INITIAL_BALANCE, RISK_FREE_RATE

def calculate_returns(portfolio_values: List[float]) -> np.ndarray:
    """Calculate daily returns from portfolio values"""
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    return returns

def calculate_cumulative_return(portfolio_values: List[float]) -> float:
    """Calculate total cumulative return"""
    return (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]

def calculate_annual_return(portfolio_values: List[float], trading_days: int = 252) -> float:
    """Calculate annualized return"""
    cumulative_return = calculate_cumulative_return(portfolio_values)
    years = len(portfolio_values) / trading_days
    if years > 0:
        annual_return = (1 + cumulative_return) ** (1 / years) - 1
    else:
        annual_return = 0
    return annual_return

def calculate_volatility(portfolio_values: List[float], trading_days: int = 252) -> float:
    """Calculate annualized volatility"""
    returns = calculate_returns(portfolio_values)
    daily_vol = np.std(returns)
    annual_vol = daily_vol * np.sqrt(trading_days)
    return annual_vol

def calculate_sharpe_ratio(portfolio_values: List[float], 
                          risk_free_rate: float = RISK_FREE_RATE,
                          trading_days: int = 252) -> float:
    """
    Calculate Sharpe ratio: (return - risk_free_rate) / volatility
    """
    annual_return = calculate_annual_return(portfolio_values, trading_days)
    annual_vol = calculate_volatility(portfolio_values, trading_days)
    
    if annual_vol == 0:
        return 0
    
    sharpe = (annual_return - risk_free_rate) / annual_vol
    return sharpe

def calculate_max_drawdown(portfolio_values: List[float]) -> float:
    """Calculate maximum drawdown"""
    portfolio_array = np.array(portfolio_values)
    cumulative_max = np.maximum.accumulate(portfolio_array)
    drawdown = (portfolio_array - cumulative_max) / cumulative_max
    max_drawdown = np.min(drawdown)
    return abs(max_drawdown)

def evaluate_model(portfolio_history: Dict, 
                  initial_balance: float = INITIAL_BALANCE) -> Dict[str, float]:
    """
    Evaluate a model's performance
    
    Args:
        portfolio_history: Dict with 'dates' and 'values' keys
        initial_balance: Starting capital
    
    Returns:
        Dictionary with all performance metrics
    """
    portfolio_values = portfolio_history['values']
    
    if len(portfolio_values) < 2:
        return {
            'cumulative_return': 0,
            'annual_return': 0,
            'volatility': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0
        }
    
    metrics = {
        'cumulative_return': calculate_cumulative_return(portfolio_values),
        'annual_return': calculate_annual_return(portfolio_values),
        'volatility': calculate_volatility(portfolio_values),
        'sharpe_ratio': calculate_sharpe_ratio(portfolio_values),
        'max_drawdown': calculate_max_drawdown(portfolio_values)
    }
    
    return metrics

def compare_models(model_results: Dict[str, Dict], 
                  baseline_results: Dict[str, Dict] = None) -> pd.DataFrame:
    """
    Compare multiple models and baselines
    
    Args:
        model_results: Dict of {model_name: portfolio_history}
        baseline_results: Dict of {baseline_name: portfolio_history}
    
    Returns:
        DataFrame with comparison metrics
    """
    all_results = {**model_results}
    if baseline_results:
        all_results.update(baseline_results)
    
    comparison = []
    for name, portfolio_history in all_results.items():
        metrics = evaluate_model(portfolio_history)
        metrics['name'] = name
        comparison.append(metrics)
    
    df = pd.DataFrame(comparison)
    df = df.set_index('name')
    
    return df

def plot_portfolio_comparison(portfolio_histories: Dict[str, Dict], 
                             save_path: str = None):
    """Plot portfolio value comparison"""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    
    for name, history in portfolio_histories.items():
        dates = history['dates']
        values = history['values']
        plt.plot(dates, values, label=name, linewidth=2)
    
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.title('Portfolio Value Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

