"""
A2C training script for stock trading RL
"""
import os
import numpy as np
import pandas as pd
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from trading_env import StockTradingEnv
from config import MODELS_DIR, LOG_DIR, LEARNING_RATE, GAMMA

def create_env(df: pd.DataFrame, tickers: list = None):
    """Create and wrap environment"""
    env = StockTradingEnv(df, tickers=tickers)
    env = Monitor(env, filename=os.path.join(LOG_DIR, 'monitor'))
    return env

def train_a2c(train_df: pd.DataFrame, 
              val_df: pd.DataFrame = None,
              tickers: list = None,
              total_timesteps: int = 200000,
              model_name: str = "a2c_stock_trading"):
    """
    Train A2C (Advantage Actor-Critic) model
    
    Args:
        train_df: Training data DataFrame
        val_df: Validation data DataFrame (optional)
        tickers: List of stock tickers
        total_timesteps: Number of training timesteps
        model_name: Name for saving the model
    
    Returns:
        Trained A2C model
    """
    print("Training A2C model...")
    
    # Create environments
    train_env = DummyVecEnv([lambda: create_env(train_df, tickers)])
    val_env = DummyVecEnv([lambda: create_env(val_df, tickers)]) if val_df is not None else None
    
    # Create A2C model
    # Try to use tensorboard if available, otherwise disable it
    try:
        import tensorboard
        tensorboard_log = os.path.join(LOG_DIR, 'a2c_tensorboard')
    except ImportError:
        tensorboard_log = None
    
    model = A2C(
        'MlpPolicy',
        train_env,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        verbose=1,
        tensorboard_log=tensorboard_log
    )
    
    # Setup callbacks
    callbacks = []
    
    if val_env is not None:
        eval_callback = EvalCallback(
            val_env,
            best_model_save_path=os.path.join(MODELS_DIR, f'{model_name}_best'),
            log_path=os.path.join(LOG_DIR, f'{model_name}_eval'),
            eval_freq=5000,
            deterministic=True,
            render=False
        )
        callbacks.append(eval_callback)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=os.path.join(MODELS_DIR, f'{model_name}_checkpoints'),
        name_prefix=model_name
    )
    callbacks.append(checkpoint_callback)
    
    # Train the model
    # Try to use progress bar if rich is available
    try:
        import rich
        use_progress_bar = True
    except ImportError:
        use_progress_bar = False
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=use_progress_bar
    )
    
    # Save final model
    model.save(os.path.join(MODELS_DIR, f'{model_name}_final'))
    print(f"A2C model saved to {os.path.join(MODELS_DIR, f'{model_name}_final')}")
    
    return model

