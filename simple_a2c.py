"""
Simple A2C implementation for stock trading
Trains a single A2C model and evaluates it
"""
import pandas as pd
import numpy as np
from data_loader import StockDataLoader
from train_rl_models import train_a2c
from trading_env import StockTradingEnv
from evaluation import evaluate_model, plot_portfolio_comparison
from baselines import djia_buy_and_hold, min_variance_portfolio
from config import DOW30_TICKERS, TRAIN_START, TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END, RESULTS_DIR
import os

def train_and_evaluate_a2c():
    """Train A2C model and evaluate performance"""
    
    print("="*80)
    print("Simple A2C Stock Trading Agent")
    print("="*80)
    
    # Step 1: Load and prepare data
    print("\n[1/4] Loading and preparing data...")
    loader = StockDataLoader(tickers=DOW30_TICKERS)
    
    # Try to load existing, otherwise download
    data_file = os.path.join('data', 'stock_data_processed.csv')
    if os.path.exists(data_file):
        print(f"Loading existing data from {data_file}")
        loader.load_from_file(data_file)
        loader.calculate_technical_indicators()
    else:
        print("Downloading fresh data (this may take a few minutes)...")
        loader.download_data(TRAIN_START, TEST_END, save=True)
        loader.calculate_technical_indicators()
        loader.processed_data.to_csv(data_file)
        print(f"Data saved to {data_file}")
    
    # Split data
    splits = loader.split_data()
    print(f"\nData splits:")
    print(f"  Training: {len(splits['train'])} days ({splits['train'].index[0]} to {splits['train'].index[-1]})")
    print(f"  Validation: {len(splits['val'])} days ({splits['val'].index[0]} to {splits['val'].index[-1]})")
    print(f"  Testing: {len(splits['test'])} days ({splits['test'].index[0]} to {splits['test'].index[-1]})")
    
    if len(splits['train']) == 0:
        raise ValueError("No training data! Please check date ranges in config.py")
    
    # Step 2: Train A2C model
    print("\n[2/4] Training A2C model...")
    print("This may take 10-30 minutes depending on your machine and timesteps")
    
    a2c_model = train_a2c(
        splits['train'],
        splits['val'],
        tickers=DOW30_TICKERS,
        total_timesteps=100000,  # Adjust this - more timesteps = better but slower
        model_name="a2c_simple"
    )
    
    print("\n✓ A2C model training complete!")
    
    # Step 3: Test on test set
    print("\n[3/4] Testing A2C model on test set...")
    test_env = StockTradingEnv(splits['test'], tickers=DOW30_TICKERS)
    
    obs, _ = test_env.reset()
    done = False
    step_count = 0
    
    while not done:
        action, _ = a2c_model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)
        step_count += 1
        
        if step_count % 50 == 0:
            print(f"  Step {step_count}/{len(splits['test'])} - Portfolio: ${info['portfolio_value']:,.2f}")
    
    a2c_history = test_env.get_portfolio_history()
    print(f"\n✓ Testing complete! Final portfolio value: ${a2c_history['values'][-1]:,.2f}")
    
    # Step 4: Evaluate and compare with baselines
    print("\n[4/4] Evaluating performance and comparing with baselines...")
    
    # Evaluate A2C
    a2c_metrics = evaluate_model(a2c_history)
    
    # Run baselines
    print("  Running Buy-and-Hold baseline...")
    buyhold_history = djia_buy_and_hold(splits['test'], DOW30_TICKERS)
    buyhold_metrics = evaluate_model(buyhold_history)
    
    print("  Running Min-Variance Portfolio baseline...")
    minvar_history = min_variance_portfolio(splits['test'], DOW30_TICKERS)
    minvar_metrics = evaluate_model(minvar_history)
    
    # Print results
    print("\n" + "="*80)
    print("PERFORMANCE METRICS")
    print("="*80)
    
    print("\nA2C Model:")
    for metric, value in a2c_metrics.items():
        print(f"  {metric:20s}: {value:.4f}")
    
    print("\nBuy-and-Hold Baseline:")
    for metric, value in buyhold_metrics.items():
        print(f"  {metric:20s}: {value:.4f}")
    
    print("\nMin-Variance Portfolio Baseline:")
    for metric, value in minvar_metrics.items():
        print(f"  {metric:20s}: {value:.4f}")
    
    # Create comparison plot
    print("\nGenerating comparison plot...")
    plot_file = os.path.join(RESULTS_DIR, 'a2c_comparison.png')
    plot_portfolio_comparison({
        'A2C': a2c_history,
        'Buy-and-Hold': buyhold_history,
        'Min-Variance': minvar_history
    }, save_path=plot_file)
    print(f"Plot saved to {plot_file}")
    
    # Save results to CSV
    import pandas as pd
    results_df = pd.DataFrame({
        'A2C': a2c_metrics,
        'Buy-and-Hold': buyhold_metrics,
        'Min-Variance': minvar_metrics
    })
    results_file = os.path.join(RESULTS_DIR, 'a2c_results.csv')
    results_df.to_csv(results_file)
    print(f"Results saved to {results_file}")
    
    print("\n" + "="*80)
    print("Summary:")
    print(f"  A2C Sharpe Ratio: {a2c_metrics['sharpe_ratio']:.4f}")
    print(f"  Buy-Hold Sharpe: {buyhold_metrics['sharpe_ratio']:.4f}")
    print(f"  Min-Var Sharpe: {minvar_metrics['sharpe_ratio']:.4f}")
    print(f"\n  A2C Annual Return: {a2c_metrics['annual_return']:.4f}")
    print(f"  Buy-Hold Return: {buyhold_metrics['annual_return']:.4f}")
    print(f"  Min-Var Return: {minvar_metrics['annual_return']:.4f}")
    print("="*80)
    
    return a2c_model, a2c_metrics

if __name__ == "__main__":
    train_and_evaluate_a2c()

