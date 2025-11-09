"""
Custom Gym environment for stock trading
State: 1 balance + N stocks * 7 features (price, shares, MACD, RSI, CCI, ADX, MFI)
Action: N continuous actions [-1, 1] for each stock
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from config import (
    INITIAL_BALANCE, TRANSACTION_COST_RATE, TURBULENCE_THRESHOLD,
    MAX_STOCK_PRICE, MAX_SHARES, DOW30_TICKERS, MAX_MFI, MAX_PE, MAX_OBV
)

class StockTradingEnv(gym.Env):
    """Simplified stock trading environment"""
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, 
                 df: pd.DataFrame,
                 tickers: list = None,
                 initial_balance: float = INITIAL_BALANCE,
                 transaction_cost: float = TRANSACTION_COST_RATE,
                 turbulence_threshold: float = TURBULENCE_THRESHOLD):
        super(StockTradingEnv, self).__init__()
        
        self.tickers = tickers or DOW30_TICKERS
        self.n_stocks = len(self.tickers)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.turbulence_threshold = turbulence_threshold
        
        # Extract data
        self.df = df.copy()
        self.dates = self.df.index
        self.n_days = len(self.dates)
        
        # Extract features
        self._extract_features()
        self._calculate_turbulence()

        # State: 1 balance + N stocks * 9 features (price, shares, MACD, RSI, CCI, ADX, MFI, OBV, PE)
        self.state_dim = 1 + self.n_stocks * 9

        # Action space: continuous [-1, 1] for each stock
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_stocks,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)

        self.reset()
    
    def _extract_features(self):
        """Extract stock prices and technical indicators"""
        self.prices = {}
        self.macd = {}
        self.rsi = {}
        self.cci = {}
        self.adx = {}
        self.mfi = {}
        self.pe = {}
        self.obv = {}

        n = len(self.df)
        
        for ticker in self.tickers:
            close_col = f"{ticker}_Close"
            
            # Extract prices
            if close_col in self.df.columns:
                self.prices[ticker] = np.asarray(self.df[close_col].values, dtype=np.float32).flatten()
            else:
                self.prices[ticker] = np.zeros(n, dtype=np.float32)
            
            # Extract technical indicators with proper defaults
            macd_col = f"{ticker}_MACD"
            if macd_col in self.df.columns:
                self.macd[ticker] = np.asarray(self.df[macd_col].values, dtype=np.float32).flatten()
            else:
                self.macd[ticker] = np.zeros(n, dtype=np.float32)
            
            rsi_col = f"{ticker}_RSI"
            if rsi_col in self.df.columns:
                self.rsi[ticker] = np.asarray(self.df[rsi_col].values, dtype=np.float32).flatten()
            else:
                self.rsi[ticker] = np.full(n, 50.0, dtype=np.float32)
            
            cci_col = f"{ticker}_CCI"
            if cci_col in self.df.columns:
                self.cci[ticker] = np.asarray(self.df[cci_col].values, dtype=np.float32).flatten()
            else:
                self.cci[ticker] = np.zeros(n, dtype=np.float32)
            
            adx_col = f"{ticker}_ADX"
            if adx_col in self.df.columns:
                self.adx[ticker] = np.asarray(self.df[adx_col].values, dtype=np.float32).flatten()
            else:
                self.adx[ticker] = np.zeros(n, dtype=np.float32)

            # prefer scaled MFI if present
            mfi_scaled_col = f"{ticker}_MFI_SCALED"
            mfi_col = f"{ticker}_MFI"
            if mfi_scaled_col in self.df.columns:
                self.mfi[ticker] = np.asarray(self.df[mfi_scaled_col].values, dtype=np.float32).flatten()
            elif mfi_col in self.df.columns:
                self.mfi[ticker] = np.asarray(self.df[mfi_col].values, dtype=np.float32).flatten()
            else:
                # neutral MFI ~50 (unscaled)
                self.mfi[ticker] = np.full(n, 50.0, dtype=np.float32)

            # OBV (On-Balance Volume)
            # prefer scaled OBV if present
            obv_scaled_col = f"{ticker}_OBV_SCALED"
            obv_col = f"{ticker}_OBV"
            if obv_scaled_col in self.df.columns:
                try:
                    self.obv[ticker] = np.asarray(self.df[obv_scaled_col].values, dtype=np.float32).flatten()
                except Exception:
                    self.obv[ticker] = np.zeros(n, dtype=np.float32)
            elif obv_col in self.df.columns:
                try:
                    self.obv[ticker] = np.asarray(self.df[obv_col].values, dtype=np.float32).flatten()
                except Exception:
                    self.obv[ticker] = np.zeros(n, dtype=np.float32)
            else:
                self.obv[ticker] = np.zeros(n, dtype=np.float32)

            # P/E column (may be a static value repeated across dates)
            # prefer scaled PE if present
            pe_scaled_col = f"{ticker}_PE_SCALED"
            pe_col = f"{ticker}_PE"
            if pe_scaled_col in self.df.columns:
                try:
                    self.pe[ticker] = np.asarray(self.df[pe_scaled_col].values, dtype=np.float32).flatten()
                except Exception:
                    self.pe[ticker] = np.full(n, 15.0, dtype=np.float32)
            elif pe_col in self.df.columns:
                try:
                    self.pe[ticker] = np.asarray(self.df[pe_col].values, dtype=np.float32).flatten()
                except Exception:
                    self.pe[ticker] = np.full(n, 15.0, dtype=np.float32)
            else:
                # sensible default P/E
                self.pe[ticker] = np.full(n, 15.0, dtype=np.float32)
    
    def _calculate_turbulence(self):
        """Calculate turbulence index (simplified)"""
        # Create price DataFrame
        price_df = pd.DataFrame({t: self.prices[t] for t in self.tickers if t in self.prices}, 
                                index=self.dates)
        
        if len(price_df) == 0:
            self.turbulence = np.zeros(self.n_days, dtype=np.float32)
            return
        
        # Calculate returns
        returns = price_df.pct_change().dropna()
        
        self.turbulence = np.zeros(self.n_days, dtype=np.float32)
        lookback = min(252, len(returns))
        
        for i in range(lookback, len(returns)):
            window = returns.iloc[i-lookback:i]
            mean_ret = window.mean()
            cov = window.cov()
            curr_ret = returns.iloc[i]
            
            try:
                diff = curr_ret - mean_ret
                inv_cov = np.linalg.pinv(cov)
                self.turbulence[i] = float(diff @ inv_cov @ diff)
            except:
                pass
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment"""
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = np.zeros(self.n_stocks, dtype=np.float32)
        self.portfolio_value_history = [self.initial_balance]
        self.action_history = []
        self.reward_history = []
        return self._get_state(), {}
    
    def _get_state(self) -> np.ndarray:
        """Get current state vector"""
        if self.current_step >= self.n_days:
            return np.zeros(self.state_dim, dtype=np.float32)
        
        state = np.zeros(self.state_dim, dtype=np.float32)
        step = self.current_step
        
        # Helper to safely convert to float and handle NaN/inf
        def safe_float(val, default=0.0):
            if isinstance(val, np.ndarray):
                val = val.item() if val.size == 1 else val.flat[0]
            elif hasattr(val, 'item'):
                val = val.item()
            
            val = float(val)
            # Replace NaN or inf with default
            if not np.isfinite(val):
                return default
            return val
        
        # Balance
        state[0] = safe_float(self.balance) / MAX_STOCK_PRICE
        
    # For each stock: price, shares, MACD, RSI, CCI, ADX, MFI, OBV, PE
        idx = 1
        for i, ticker in enumerate(self.tickers):
            # Get values with NaN handling
            price_val = safe_float(self.prices[ticker][step], 0.0)
            shares_val = safe_float(self.shares_held[i], 0.0)
            macd_val = safe_float(self.macd[ticker][step], 0.0)
            rsi_val = safe_float(self.rsi[ticker][step], 50.0)
            cci_val = safe_float(self.cci[ticker][step], 0.0)
            adx_val = safe_float(self.adx[ticker][step], 0.0)
            mfi_val = safe_float(self.mfi[ticker][step], 50.0)
            obv_val = safe_float(self.obv[ticker][step], 0.0)
            pe_val = safe_float(self.pe[ticker][step], 15.0)

            # Normalize and store (handle zero prices)
            state[idx] = price_val / MAX_STOCK_PRICE if price_val > 0 else 0.0
            state[idx+1] = shares_val / MAX_SHARES
            state[idx+2] = macd_val / 1000.0
            state[idx+3] = rsi_val / 100.0
            state[idx+4] = cci_val / 1000.0
            state[idx+5] = adx_val / 100.0
            # normalize MFI (if already scaled use directly, otherwise divide by MAX_MFI)
            if np.isfinite(mfi_val) and abs(mfi_val) <= 1.0001:
                state[idx+6] = mfi_val
            else:
                state[idx+6] = mfi_val / MAX_MFI if np.isfinite(mfi_val) else 0.5

            # normalize OBV (if scaled, values will be in [-1,1]; otherwise divide by MAX_OBV)
            if np.isfinite(obv_val) and abs(obv_val) <= 1.0001:
                state[idx+7] = obv_val
            else:
                state[idx+7] = obv_val / MAX_OBV if np.isfinite(obv_val) else 0.0

            # normalize PE (if scaled, values will be <=1; otherwise divide by MAX_PE)
            if np.isfinite(pe_val) and abs(pe_val) <= 1.0001:
                state[idx+8] = pe_val
            else:
                state[idx+8] = (pe_val / MAX_PE) if np.isfinite(pe_val) else (15.0 / MAX_PE)
            idx += 9
        
        # Final check: replace any remaining NaN/inf
        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        
        return state.astype(np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step"""
        action = np.clip(action, -1, 1)
        
        # Halt trading if turbulence too high
        if self.turbulence[self.current_step] > self.turbulence_threshold:
            action = np.zeros_like(action)
        
        # Get current prices (handle NaN/zero)
        prices = np.array([max(self.prices[t][self.current_step], 0.001) for t in self.tickers], dtype=np.float32)
        prices = np.nan_to_num(prices, nan=0.001, posinf=1e6, neginf=0.001)
        
        # Portfolio value before trades
        portfolio_before = self.balance + np.sum(self.shares_held * prices)
        
        # Execute trades
        for i in range(self.n_stocks):
            price = float(prices[i])
            
            # Skip if invalid price
            if price <= 0 or not np.isfinite(price):
                continue
            
            if action[i] > 0:  # Buy
                amount = abs(action[i]) * self.balance
                if amount > 0 and amount <= self.balance:
                    shares = amount / (price * (1 + self.transaction_cost))
                    if shares > 0 and np.isfinite(shares):
                        self.balance -= amount
                        self.shares_held[i] += shares
            
            elif action[i] < 0:  # Sell
                shares = abs(action[i]) * self.shares_held[i]
                if shares > 0 and shares <= self.shares_held[i]:
                    revenue = shares * price * (1 - self.transaction_cost)
                    if np.isfinite(revenue):
                        self.balance += revenue
                        self.shares_held[i] = max(0, self.shares_held[i] - shares)
        
        # Portfolio value after trades
        portfolio_after = self.balance + np.sum(self.shares_held * prices)
        
        # Reward = change in portfolio value
        reward = portfolio_after - portfolio_before
        
        # Update
        self.current_step += 1
        self.portfolio_value_history.append(portfolio_after)
        self.action_history.append(action.copy())
        self.reward_history.append(reward)
        
        # Done if reached end
        done = self.current_step >= self.n_days
        
        info = {
            'portfolio_value': portfolio_after,
            'balance': self.balance,
            'turbulence': self.turbulence[self.current_step - 1],
            'date': self.dates[self.current_step - 1] if self.current_step > 0 else self.dates[0]
        }
        
        return self._get_state(), reward, done, False, info
    
    def render(self):
        """Render environment state"""
        print(f"Step: {self.current_step}, Balance: ${self.balance:,.2f}, "
              f"Portfolio: ${self.portfolio_value_history[-1]:,.2f}")
    
    def get_portfolio_history(self) -> Dict:
        """Get portfolio history"""
        # Skip initial balance value to match dates length
        # Dates correspond to each step, so we need one value per date
        if len(self.portfolio_value_history) > len(self.dates):
            # If we have one extra value (initial), skip it
            values = self.portfolio_value_history[1:]
        else:
            values = self.portfolio_value_history
        
        # Ensure dates and values have same length
        min_len = min(len(self.dates), len(values))
        return {
            'dates': list(self.dates[:min_len]),
            'values': values[:min_len],
            'actions': self.action_history[:min_len],
            'rewards': self.reward_history[:min_len]
        }
