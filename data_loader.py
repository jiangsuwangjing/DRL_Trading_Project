"""
Data loading and preprocessing module
Downloads stock data and calculates technical indicators
"""
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from typing import List, Tuple, Dict
from config import DOW30_TICKERS, TRAIN_START, TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END, DATA_DIR, MAX_MFI, MAX_PE, MAX_OBV
import os

class StockDataLoader:
    """Loads and preprocesses stock data with technical indicators"""
    
    def __init__(self, tickers: List[str] = None):
        self.tickers = tickers or DOW30_TICKERS
        self.data = None
        self.processed_data = None
        
    def download_data(self, start_date: str, end_date: str, save: bool = True) -> pd.DataFrame:
        """Download historical stock data from Yahoo Finance"""
        print(f"Downloading data from {start_date} to {end_date}...")
        
        all_data = {}
        for ticker in self.tickers:
            try:
                print(f"  Fetching {ticker}...")
                stock = yf.Ticker(ticker)
                df = stock.history(start=start_date, end=end_date)
                if not df.empty:
                    all_data[ticker] = df
                else:
                    print(f"  Warning: No data for {ticker}")
            except Exception as e:
                print(f"  Error downloading {ticker}: {e}")
        
        # Combine all tickers into a single multi-index DataFrame
        if not all_data:
            raise ValueError("No data downloaded!")
        
        # Align dates across all stocks
        dates = None
        for ticker, df in all_data.items():
            if dates is None:
                dates = set(df.index)
            else:
                dates = dates.intersection(set(df.index))
        
        dates = sorted(list(dates))
        
        # Create combined DataFrame using pd.concat to avoid fragmentation
        data_dict = {}
        for ticker in self.tickers:
            if ticker in all_data:
                df = all_data[ticker].loc[dates]
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    data_dict[f"{ticker}_{col}"] = df[col].values
        
        # Create DataFrame from dictionary at once
        combined_data = pd.DataFrame(data_dict, index=dates)
        combined_data.index = pd.to_datetime(combined_data.index).tz_localize(None)  # Ensure timezone-naive
        
        if save:
            filepath = os.path.join(DATA_DIR, f"stock_data_{start_date}_{end_date}.csv")
            combined_data.to_csv(filepath)
            print(f"Data saved to {filepath}")
        
        self.data = combined_data
        return combined_data
    
    def calculate_technical_indicators(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """Calculate technical indicators: MACD, RSI, CCI, ADX"""
        if df is None:
            df = self.data.copy()
        
        processed_data = df.copy()
        
        print("Calculating technical indicators...")
        # Collect all new columns in a dictionary first
        new_cols = {}
        
        for ticker in self.tickers:
            close_col = f"{ticker}_Close"
            high_col = f"{ticker}_High"
            low_col = f"{ticker}_Low"
            
            if close_col not in df.columns:
                continue
            
            # MACD (Moving Average Convergence Divergence)
            macd = ta.macd(df[close_col], fast=12, slow=26, signal=9)
            if macd is not None and isinstance(macd, pd.DataFrame):
                new_cols[f"{ticker}_MACD"] = macd[f'MACD_12_26_9'].values
                new_cols[f"{ticker}_MACD_signal"] = macd[f'MACDs_12_26_9'].values
            else:
                new_cols[f"{ticker}_MACD"] = np.zeros(len(df))
            
            # RSI (Relative Strength Index)
            rsi = ta.rsi(df[close_col], length=14)
            new_cols[f"{ticker}_RSI"] = rsi.values if rsi is not None else np.full(len(df), 50.0)
            
            # CCI (Commodity Channel Index)
            cci = ta.cci(df[high_col], df[low_col], df[close_col], length=20)
            new_cols[f"{ticker}_CCI"] = cci.values if cci is not None else np.zeros(len(df))
            
            # ADX (Average Directional Index)
            adx = ta.adx(df[high_col], df[low_col], df[close_col], length=14)
            if adx is not None and isinstance(adx, pd.DataFrame):
                new_cols[f"{ticker}_ADX"] = adx['ADX_14'].values
            else:
                new_cols[f"{ticker}_ADX"] = np.zeros(len(df))
            # MFI (Money Flow Index) - volume-weighted momentum indicator (0-100)
            try:
                # pandas_ta.mfi expects high, low, close, volume
                mfi = ta.mfi(df[high_col], df[low_col], df[close_col], df[f"{ticker}_Volume"], length=14)
            except Exception:
                mfi = None

            mfi_arr = mfi.values if mfi is not None else np.full(len(df), 50.0)
            new_cols[f"{ticker}_MFI"] = mfi_arr
            # scaled MFI (0-1)
            try:
                new_cols[f"{ticker}_MFI_SCALED"] = (mfi_arr / MAX_MFI).astype(float)
            except Exception:
                new_cols[f"{ticker}_MFI_SCALED"] = mfi_arr.astype(float) / MAX_MFI

            # OBV (On-Balance Volume) - momentum indicator using volume flow
            try:
                # pandas_ta.obv typically expects close, volume
                obv = ta.obv(df[close_col], df[f"{ticker}_Volume"])
            except Exception:
                obv = None

            obv_arr = obv.values if obv is not None else np.zeros(len(df))
            new_cols[f"{ticker}_OBV"] = obv_arr
            # scaled OBV per-ticker to [-1,1] by dividing by max absolute
            try:
                abs_max = np.nanmax(np.abs(obv_arr)) if obv_arr is not None else 0.0
                if not np.isfinite(abs_max) or abs_max == 0:
                    scaled_obv = np.zeros(len(df))
                else:
                    scaled_obv = obv_arr / float(abs_max)
            except Exception:
                scaled_obv = np.zeros(len(df))
            new_cols[f"{ticker}_OBV_SCALED"] = scaled_obv

            # P/E ratio (trailing P/E) - fundamental data is mostly static across our time series
            try:
                info = yf.Ticker(ticker).info
                pe_val = info.get('trailingPE') if isinstance(info, dict) else None
            except Exception:
                pe_val = None

            # sensible default if PE not available
            default_pe = 15.0
            try:
                if pe_val is None or (isinstance(pe_val, float) and np.isnan(pe_val)):
                    pe_val = default_pe
                # create a column with the constant PE value across dates
                pe_arr = np.full(len(df), float(pe_val))
            except Exception:
                pe_arr = np.full(len(df), default_pe)

            new_cols[f"{ticker}_PE"] = pe_arr
            # scaled PE
            new_cols[f"{ticker}_PE_SCALED"] = (pe_arr / MAX_PE).astype(float)
            # P/E ratio (trailing P/E) - fundamental data is mostly static across our time series
            # We fetch it via yfinance as a fallback; if unavailable we fill with a sensible default
            try:
                info = yf.Ticker(ticker).info
                pe_val = info.get('trailingPE') if isinstance(info, dict) else None
            except Exception:
                pe_val = None

            # sensible default if PE not available
            default_pe = 15.0
            try:
                if pe_val is None or (isinstance(pe_val, float) and np.isnan(pe_val)):
                    pe_val = default_pe
                # create a column with the constant PE value across dates
                new_cols[f"{ticker}_PE"] = np.full(len(df), float(pe_val))
            except Exception:
                new_cols[f"{ticker}_PE"] = np.full(len(df), default_pe)
        
        # Add all new columns at once
        indicator_df = pd.DataFrame(new_cols, index=df.index)
        processed_data = pd.concat([processed_data, indicator_df], axis=1)
        
        # Forward fill missing values
        processed_data = processed_data.ffill().bfill().fillna(0)
        
        self.processed_data = processed_data
        return processed_data
    
    def calculate_turbulence_index(self, prices: pd.DataFrame, lookback: int = 252) -> pd.Series:
        """
        Calculate turbulence index as per equation (3) in the paper
        Turbulence = (r - r_mean)^T * Cov^-1 * (r - r_mean)
        """
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        turbulence = []
        for i in range(lookback, len(returns)):
            window_returns = returns.iloc[i-lookback:i]
            
            # Mean returns
            mean_returns = window_returns.mean()
            
            # Covariance matrix
            cov_matrix = window_returns.cov()
            
            # Current returns
            current_returns = returns.iloc[i]
            
            # Calculate turbulence
            diff = current_returns - mean_returns
            try:
                inv_cov = np.linalg.pinv(cov_matrix)  # Pseudo-inverse for stability
                turb = diff @ inv_cov @ diff
                turbulence.append(turb)
            except:
                turbulence.append(0)
        
        # Pad with zeros for the first lookback days
        turbulence = [0] * lookback + turbulence
        
        return pd.Series(turbulence, index=prices.index)
    
    def split_data(self, processed_df: pd.DataFrame = None) -> Dict[str, pd.DataFrame]:
        """Split data into train, validation, and test sets"""
        if processed_df is None:
            processed_df = self.processed_data
        
        # Ensure index is DatetimeIndex and timezone-naive for consistent comparison
        if not isinstance(processed_df.index, pd.DatetimeIndex):
            processed_df.index = pd.to_datetime(processed_df.index)
        
        # Remove timezone if present
        if hasattr(processed_df.index, 'tz') and processed_df.index.tz is not None:
            processed_df.index = processed_df.index.tz_convert(None)
        
        # Convert date strings to Timestamps (ensure timezone-naive)
        # Use pd.Timestamp to create naive timestamps directly
        train_start = pd.Timestamp(TRAIN_START)
        train_end = pd.Timestamp(TRAIN_END)
        val_start = pd.Timestamp(VAL_START)
        val_end = pd.Timestamp(VAL_END)
        test_start = pd.Timestamp(TEST_START)
        test_end = pd.Timestamp(TEST_END)
        
        splits = {
            'train': processed_df[(processed_df.index >= train_start) & (processed_df.index <= train_end)],
            'val': processed_df[(processed_df.index >= val_start) & (processed_df.index <= val_end)],
            'test': processed_df[(processed_df.index >= test_start) & (processed_df.index <= test_end)]
        }
        
        return splits
    
    def load_from_file(self, filepath: str) -> pd.DataFrame:
        """Load pre-downloaded data from CSV"""
        self.data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        
        # Ensure index is DatetimeIndex and timezone-naive
        if not isinstance(self.data.index, pd.DatetimeIndex):
            self.data.index = pd.to_datetime(self.data.index)
        
        # Remove timezone if present - use tz_convert for timezone-aware, tz_localize for naive
        if hasattr(self.data.index, 'tz') and self.data.index.tz is not None:
            self.data.index = self.data.index.tz_convert(None)
        elif hasattr(self.data.index, 'tz'):
            # Already timezone-naive, ensure it stays that way
            pass
        
        # Final check: if still timezone-aware somehow, convert to naive
        try:
            if self.data.index.tz is not None:
                self.data.index = pd.to_datetime(self.data.index).tz_localize(None)
        except:
            # If tz attribute doesn't exist or can't check, recreate as naive
            self.data.index = pd.to_datetime(self.data.index.strftime('%Y-%m-%d'))
        
        return self.data

