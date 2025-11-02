# Deep Reinforcement Learning for Automated Stock Trading - A2C Implementation

This project implements an A2C (Advantage Actor-Critic) reinforcement learning agent for automated stock trading using Dow Jones 30 stocks.

## Overview

The project implements:
- **Custom Gym Environment** for stock trading with 181-dimensional state space
- **A2C Algorithm**: Advantage Actor-Critic for continuous action space trading
- **Technical Indicators**: MACD, RSI, CCI, ADX for each stock
- **Baseline Comparisons**: DJIA Buy-and-Hold and Min-Variance Portfolio
- **Performance Metrics**: Cumulative return, annual return, volatility, Sharpe ratio, max drawdown

## Project Structure

```
.
├── config.py              # Configuration parameters
├── data_loader.py         # Data downloading and preprocessing
├── trading_env.py         # Custom Gym trading environment
├── train_rl_models.py     # A2C training function
├── simple_a2c.py          # Main A2C training and evaluation script
├── baselines.py           # Baseline strategies
├── evaluation.py          # Performance metrics and evaluation
├── main.py                # Main execution script (calls simple_a2c)
├── requirements.txt       # Python dependencies
└── README.md              # This file

Directories:
├── data/                  # Stock data (downloaded)
├── models/                # Trained model checkpoints
├── results/               # Evaluation results and plots
└── logs/                  # Training logs and TensorBoard files
```

## Installation

### Python Version Requirement

**Important**: This project requires Python 3.8-3.12. PyTorch (required by stable-baselines3) does not currently support Python 3.13.

If you have Python 3.13, you have two options:

1. **Recommended**: Use Python 3.11 or 3.12
   ```bash
   # With uv, create a project with Python 3.12
   uv python install 3.12
   uv venv --python 3.12
   source .venv/bin/activate  # On macOS/Linux
   # or
   .venv\Scripts\activate  # On Windows
   ```

2. **Alternative**: Use conda to create an environment with Python 3.12
   ```bash
   conda create -n stock_rl python=3.12
   conda activate stock_rl
   ```

### Using uv (Recommended - Faster and Better Dependency Resolution)

1. Install uv if you haven't already:
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with Homebrew
brew install uv

# Or with pip
pip install uv
```

2. **Ensure you're using Python 3.8-3.12**, then install dependencies:
```bash
uv pip install -r requirements.txt
```

Or use uv's project mode:
```bash
uv python install 3.12  # Ensure Python 3.12 is available
uv sync
```

### Using pip (Alternative)

1. Install dependencies:
```bash
pip install -r requirements.txt
```

Note: You may encounter dependency conflicts with pip. If so, use uv instead.

3. (Optional) For advanced technical indicators, install TA-Lib:
```bash
# macOS
brew install ta-lib
pip install TA-Lib

# Linux
# See TA-Lib installation instructions for your distribution
```

## Usage

### Quick Start

Run the complete A2C training and evaluation pipeline:
```bash
python main.py
```

Or directly:
```bash
python simple_a2c.py
```

This will:
1. Download/prepare stock data for Dow Jones 30 stocks
2. Train an A2C model on the training set
3. Validate on the validation set (saves best model)
4. Test on the test set
5. Compare with baseline strategies (Buy-and-Hold, Min-Variance)
6. Generate performance metrics and visualizations

### Step-by-Step Usage

#### 1. Load and Prepare Data

```python
from data_loader import StockDataLoader
from config import DOW30_TICKERS, TRAIN_START, TEST_END

loader = StockDataLoader(tickers=DOW30_TICKERS)
loader.download_data(TRAIN_START, TEST_END)
loader.calculate_technical_indicators()
splits = loader.split_data()
```

#### 2. Train A2C Model

```python
from train_rl_models import train_a2c

# Train on training set, validate on validation set
a2c_model = train_a2c(splits['train'], splits['val'], tickers=DOW30_TICKERS)
```

#### 3. Test the Model

```python
from trading_env import StockTradingEnv

test_env = StockTradingEnv(splits['test'], tickers=DOW30_TICKERS)
obs, _ = test_env.reset()
done = False

while not done:
    action, _ = a2c_model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = test_env.step(action)

portfolio_history = test_env.get_portfolio_history()
```

#### 4. Evaluate Performance

```python
from evaluation import evaluate_model

metrics = evaluate_model(portfolio_history)
print(metrics)
```

#### 5. Compare with Baselines

```python
from baselines import djia_buy_and_hold, min_variance_portfolio

buyhold = djia_buy_and_hold(splits['test'], DOW30_TICKERS)
minvar = min_variance_portfolio(splits['test'], DOW30_TICKERS)

comparison = compare_models(
    {'A2C': portfolio_history},
    {'Buy-and-Hold': buyhold, 'Min-Variance': minvar}
)
```

## Configuration

Edit `config.py` to adjust:
- Stock tickers
- Date ranges for train/validation/test
- Trading parameters (initial balance, transaction costs, etc.)
- RL hyperparameters (learning rate, batch size, etc.)
- Rolling window and validation periods

## Key Features

### State Space (181 dimensions)
- 1 balance
- 30 stocks × 6 features (price, shares, MACD, RSI, CCI, ADX)

### Action Space
- 30 continuous actions [-1, 1] per stock
- -1 = sell all, 0 = hold, +1 = buy all

### Reward Function
- Change in portfolio value minus transaction costs

### A2C Training
- Train on historical data (2014-2019)
- Validate on validation set (2020)
- Test on out-of-sample data (2021-2024)
- Model checkpoints saved during training

## Performance Metrics

- **Cumulative Return**: Total return over period
- **Annual Return**: Annualized return
- **Volatility**: Annualized standard deviation
- **Sharpe Ratio**: Risk-adjusted return (used for model selection)
- **Max Drawdown**: Largest peak-to-trough decline

## Notes

- Training time: ~10-30 minutes for 100k timesteps (depends on your machine)
- For faster testing, reduce `total_timesteps` in `simple_a2c.py` (default: 100000)
- For better performance, increase to 200000+ timesteps
- Models are saved automatically in the `models/` directory
- Best model (based on validation) is saved as `a2c_simple_best`
- TensorBoard logs are saved in `logs/` for visualization

## Troubleshooting

1. **Data download fails**: Check internet connection and yfinance API status
2. **Memory issues**: Reduce date range or number of tickers
3. **Training too slow**: Reduce `total_timesteps` or use shorter validation periods
4. **Missing indicators**: Ensure pandas-ta is installed correctly

## References

- Algorithm: A2C (Advantage Actor-Critic) from Stable-Baselines3
- Inspired by: "Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy" (ICAIF 2020)
- This implementation focuses on A2C only (simplified from the ensemble approach)

## License

This implementation is for research and educational purposes.

