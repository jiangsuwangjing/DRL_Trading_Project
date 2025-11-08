"""
Configuration file for stock trading RL project
"""
import os

# Data configuration
DOW30_TICKERS = [
    'AAPL', 'MSFT', 'UNH', 'GS', 'HD', 'MCD', 'CAT', 'TRV', 'V', 'AXP',
    'AMGN', 'BA', 'CVX', 'HON', 'IBM', 'JNJ', 'JPM', 'NKE', 'PG', 'CRM',
    'CSCO', 'DIS', 'DOW', 'INTC', 'MRK', 'VZ', 'WBA', 'WMT', 'AMZN', 'TSLA'
]

# Date ranges
TRAIN_START = '2009-01-01'
TRAIN_END = '2015-10-01'
VAL_START = '2015-10-01'
VAL_END = '2016-01-01'
# TEST_START = '2021-01-01'
# TEST_END = '2024-12-31'
TEST_START = '2016-01-01'
TEST_END = '2020-05-08'

# Trading parameters
INITIAL_BALANCE = 1000000  # $1M initial capital
TRANSACTION_COST_RATE = 0.001  # 0.1% transaction cost
RISK_FREE_RATE = 0.02  # 2% annual risk-free rate
TURBULENCE_THRESHOLD = 140  # Turbulence index threshold

# Environment parameters
MAX_STOCK_PRICE = 1e6  # Normalization factor
MAX_SHARES = 1e6  # Normalization factor
WINDOW_SIZE = 1  # Number of days to look back (1 for daily data)

# RL Training parameters
ROLLING_WINDOW_MONTHS = 3  # Retrain every 3 months
VALIDATION_MONTHS = 3  # Validate for 3 months before selection

# Model parameters
LEARNING_RATE = 3e-4
BATCH_SIZE = 64
GAMMA = 0.99  # Discount factor
TAU = 0.005  # Soft update coefficient for DDPG

# Directory paths
DATA_DIR = 'data'
MODELS_DIR = 'models'
RESULTS_DIR = 'results'
LOG_DIR = 'logs'

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)

