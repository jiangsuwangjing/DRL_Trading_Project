"""
Quick setup test script
Verifies that all dependencies are installed and basic functionality works
"""
import sys

def test_imports():
    """Test that all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import numpy
        print("  ✓ numpy")
    except ImportError:
        print("  ✗ numpy - Run: uv pip install numpy or pip install numpy")
        return False
    
    try:
        import pandas
        print("  ✓ pandas")
    except ImportError:
        print("  ✗ pandas - Run: uv pip install pandas or pip install pandas")
        return False
    
    try:
        import gymnasium
        print("  ✓ gymnasium")
    except ImportError:
        print("  ✗ gymnasium - Run: uv pip install gymnasium or pip install gymnasium")
        return False
    
    try:
        import stable_baselines3
        print("  ✓ stable-baselines3")
    except ImportError:
        print("  ✗ stable-baselines3 - Run: uv pip install stable-baselines3 or pip install stable-baselines3")
        return False
    
    try:
        import yfinance
        print("  ✓ yfinance")
    except ImportError:
        print("  ✗ yfinance - Run: uv pip install yfinance or pip install yfinance")
        return False
    
    try:
        import pandas_ta
        print("  ✓ pandas-ta")
    except ImportError:
        print("  ✗ pandas-ta - Run: uv pip install pandas-ta or pip install pandas-ta")
        return False
    
    try:
        from scipy.optimize import minimize
        print("  ✓ scipy")
    except ImportError:
        print("  ✗ scipy - Run: uv pip install scipy or pip install scipy")
        return False
    
    return True

def test_module_imports():
    """Test that our modules can be imported"""
    print("\nTesting project modules...")
    
    try:
        from config import DOW30_TICKERS, INITIAL_BALANCE
        print(f"  ✓ config (found {len(DOW30_TICKERS)} tickers)")
    except Exception as e:
        print(f"  ✗ config - {e}")
        return False
    
    try:
        from data_loader import StockDataLoader
        print("  ✓ data_loader")
    except Exception as e:
        print(f"  ✗ data_loader - {e}")
        return False
    
    try:
        from trading_env import StockTradingEnv
        print("  ✓ trading_env")
    except Exception as e:
        print(f"  ✗ trading_env - {e}")
        return False
    
    try:
        from train_rl_models import train_a2c, train_ppo, train_ddpg
        print("  ✓ train_rl_models")
    except Exception as e:
        print(f"  ✗ train_rl_models - {e}")
        return False
    
    try:
        from evaluation import evaluate_model, calculate_sharpe_ratio
        print("  ✓ evaluation")
    except Exception as e:
        print(f"  ✗ evaluation - {e}")
        return False
    
    try:
        from simple_a2c import train_and_evaluate_a2c
        print("  ✓ simple_a2c")
    except Exception as e:
        print(f"  ✗ simple_a2c - {e}")
        return False
    
    try:
        from baselines import djia_buy_and_hold, min_variance_portfolio
        print("  ✓ baselines")
    except Exception as e:
        print(f"  ✗ baselines - {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality with dummy data"""
    print("\nTesting basic functionality...")
    
    try:
        import pandas as pd
        import numpy as np
        from trading_env import StockTradingEnv
        
        # Create dummy data
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        dummy_df = pd.DataFrame(index=dates)
        
        from config import DOW30_TICKERS
        for ticker in DOW30_TICKERS[:5]:  # Just test with 5 stocks
            dummy_df[f"{ticker}_Close"] = 100 + np.random.randn(10)
            dummy_df[f"{ticker}_High"] = dummy_df[f"{ticker}_Close"] * 1.01
            dummy_df[f"{ticker}_Low"] = dummy_df[f"{ticker}_Close"] * 0.99
            dummy_df[f"{ticker}_MACD"] = np.random.randn(10)
            dummy_df[f"{ticker}_RSI"] = 50 + np.random.randn(10)
            dummy_df[f"{ticker}_CCI"] = np.random.randn(10)
            dummy_df[f"{ticker}_ADX"] = np.random.randn(10)
        
        # Test environment creation
        env = StockTradingEnv(dummy_df, tickers=DOW30_TICKERS[:5])
        print("  ✓ Environment creation")
        
        # Test reset
        obs, info = env.reset()
        print(f"  ✓ Environment reset (state shape: {obs.shape})")
        
        # Test step
        action = np.random.uniform(-0.5, 0.5, size=5)
        obs, reward, done, truncated, info = env.step(action)
        print(f"  ✓ Environment step (reward: {reward:.2f})")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Basic functionality test failed - {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*60)
    print("Setup Test Script")
    print("="*60)
    
    all_passed = True
    
    if not test_imports():
        print("\n❌ Some required packages are missing!")
        print("Please install them with: uv pip install -r requirements.txt")
        print("(Or use: pip install -r requirements.txt)")
        all_passed = False
    
    if not test_module_imports():
        print("\n❌ Some project modules failed to import!")
        all_passed = False
    
    if all_passed and test_basic_functionality():
        print("\n" + "="*60)
        print("✓ All tests passed! You're ready to run the main pipeline.")
        print("="*60)
        print("\nNext steps:")
        print("1. Run: python main.py (for full pipeline)")
        print("2. Or start with smaller tests in test.ipynb")
    else:
        print("\n" + "="*60)
        print("❌ Some tests failed. Please fix the issues above.")
        print("="*60)
        sys.exit(1)

