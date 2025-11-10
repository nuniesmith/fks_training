"""
22D Feature Extractor for PPO Trading

Extracts 22-dimensional feature vector from market data for PPO meta-learning.
Features include technical indicators, regime detection, and microstructure.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from loguru import logger
import talib


class FKSFeatureExtractor:
    """Extract 22D feature vector for PPO trading"""
    
    def __init__(self, normalize: bool = True):
        """Initialize feature extractor
        
        Args:
            normalize: Whether to normalize features
        """
        self.normalize = normalize
        self.feature_stats = {}  # For normalization
        
    def extract_features(
        self,
        data: pd.DataFrame,
        current_idx: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """Extract 22D feature vector from market data
        
        Feature breakdown:
        1. Price features (3): close, high, low (normalized)
        2. Volume features (2): volume, volume_ma_ratio
        3. Technical indicators (10): RSI, MACD, MACD_signal, BB_upper, BB_lower, 
           ATR, ADX, CCI, Stochastic, Williams %R
        4. Moving averages (3): SMA_20, SMA_50, EMA_12
        5. Regime indicators (4): trend_regime, volatility_regime, momentum_regime, volume_regime
        6. Total: 22 features
        
        Args:
            data: DataFrame with OHLCV data
            current_idx: Current index (if None, uses last row)
        
        Returns:
            22D feature vector (np.ndarray) or None if insufficient data
        """
        if data is None or len(data) < 50:
            logger.warning("Insufficient data for feature extraction")
            return None
        
        if current_idx is None:
            current_idx = len(data) - 1
        
        if current_idx < 50:
            logger.warning(f"Current index {current_idx} too small for feature extraction")
            return None
        
        # Get window of data for calculations
        window_data = data.iloc[max(0, current_idx - 50):current_idx + 1]
        current_data = data.iloc[current_idx]
        
        features = []
        
        # 1. Price features (3)
        close = current_data.get("Close", current_data.get("close", 0))
        high = current_data.get("High", current_data.get("high", 0))
        low = current_data.get("Low", current_data.get("low", 0))
        
        # Normalize prices by recent average
        price_avg = window_data["Close"].mean() if "Close" in window_data.columns else window_data["close"].mean()
        if price_avg > 0:
            features.append((close - price_avg) / price_avg)  # Normalized close
            features.append((high - price_avg) / price_avg)  # Normalized high
            features.append((low - price_avg) / price_avg)  # Normalized low
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # 2. Volume features (2)
        volume = current_data.get("Volume", current_data.get("volume", 0))
        volume_ma = window_data["Volume"].rolling(20).mean().iloc[-1] if "Volume" in window_data.columns else window_data["volume"].rolling(20).mean().iloc[-1]
        if volume_ma > 0:
            features.append(volume / volume_ma)  # Volume ratio
        else:
            features.append(1.0)
        features.append(np.log(volume + 1) / 10.0)  # Log volume (normalized)
        
        # 3. Technical indicators (10)
        close_series = window_data["Close"] if "Close" in window_data.columns else window_data["close"]
        high_series = window_data["High"] if "High" in window_data.columns else window_data["high"]
        low_series = window_data["Low"] if "Low" in window_data.columns else window_data["low"]
        volume_series = window_data["Volume"] if "Volume" in window_data.columns else window_data["volume"]
        
        # RSI (14)
        rsi = talib.RSI(close_series.values, timeperiod=14)
        features.append(rsi[-1] / 100.0 if not np.isnan(rsi[-1]) else 0.5)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(close_series.values)
        features.append(macd[-1] / close if not np.isnan(macd[-1]) and close > 0 else 0.0)
        features.append(macd_signal[-1] / close if not np.isnan(macd_signal[-1]) and close > 0 else 0.0)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close_series.values, timeperiod=20)
        if not np.isnan(bb_upper[-1]) and close > 0:
            features.append((bb_upper[-1] - close) / close)  # Distance from upper band
            features.append((close - bb_lower[-1]) / close)  # Distance from lower band
        else:
            features.extend([0.0, 0.0])
        
        # ATR (14)
        atr = talib.ATR(high_series.values, low_series.values, close_series.values, timeperiod=14)
        features.append(atr[-1] / close if not np.isnan(atr[-1]) and close > 0 else 0.0)
        
        # ADX (14)
        adx = talib.ADX(high_series.values, low_series.values, close_series.values, timeperiod=14)
        features.append(adx[-1] / 100.0 if not np.isnan(adx[-1]) else 0.0)
        
        # CCI (14)
        cci = talib.CCI(high_series.values, low_series.values, close_series.values, timeperiod=14)
        features.append(cci[-1] / 100.0 if not np.isnan(cci[-1]) else 0.0)  # Normalize to [-1, 1] range
        
        # Stochastic
        slowk, slowd = talib.STOCH(high_series.values, low_series.values, close_series.values)
        features.append(slowk[-1] / 100.0 if not np.isnan(slowk[-1]) else 0.5)
        
        # Williams %R
        willr = talib.WILLR(high_series.values, low_series.values, close_series.values, timeperiod=14)
        features.append(willr[-1] / -100.0 if not np.isnan(willr[-1]) else 0.0)  # Convert to [0, 1]
        
        # 4. Moving averages (3)
        sma_20 = talib.SMA(close_series.values, timeperiod=20)
        sma_50 = talib.SMA(close_series.values, timeperiod=50)
        ema_12 = talib.EMA(close_series.values, timeperiod=12)
        
        features.append((close - sma_20[-1]) / close if not np.isnan(sma_20[-1]) and close > 0 else 0.0)
        features.append((close - sma_50[-1]) / close if not np.isnan(sma_50[-1]) and close > 0 else 0.0)
        features.append((close - ema_12[-1]) / close if not np.isnan(ema_12[-1]) and close > 0 else 0.0)
        
        # 5. Regime indicators (4)
        features.append(self._trend_regime(window_data, current_idx))
        features.append(self._volatility_regime(window_data, current_idx))
        features.append(self._momentum_regime(window_data, current_idx))
        features.append(self._volume_regime(window_data, current_idx))
        
        # Convert to numpy array
        feature_vector = np.array(features, dtype=np.float32)
        
        # Handle NaN/inf values
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Normalize if requested
        if self.normalize:
            feature_vector = self._normalize_features(feature_vector)
        
        # Verify we have exactly 22 features
        if len(feature_vector) != 22:
            logger.error(f"Expected 22 features, got {len(feature_vector)}")
            # Pad or truncate if needed
            if len(feature_vector) < 22:
                feature_vector = np.pad(feature_vector, (0, 22 - len(feature_vector)), mode='constant')
            else:
                feature_vector = feature_vector[:22]
        
        return feature_vector
    
    def _trend_regime(self, data: pd.DataFrame, current_idx: int) -> float:
        """Detect trend regime (1.0 = strong uptrend, -1.0 = strong downtrend)"""
        close_series = data["Close"] if "Close" in data.columns else data["close"]
        
        if len(close_series) < 20:
            return 0.0
        
        # Simple trend: compare short MA to long MA
        short_ma = close_series.rolling(10).mean().iloc[-1]
        long_ma = close_series.rolling(20).mean().iloc[-1]
        current_price = close_series.iloc[-1]
        
        if long_ma > 0:
            # Trend strength based on MA crossover and price position
            ma_diff = (short_ma - long_ma) / long_ma
            price_diff = (current_price - long_ma) / long_ma
            trend = (ma_diff + price_diff) / 2.0
            return np.clip(trend * 10, -1.0, 1.0)  # Scale and clip
        return 0.0
    
    def _volatility_regime(self, data: pd.DataFrame, current_idx: int) -> float:
        """Detect volatility regime (1.0 = high volatility, -1.0 = low volatility)"""
        close_series = data["Close"] if "Close" in data.columns else data["close"]
        
        if len(close_series) < 20:
            return 0.0
        
        # Calculate volatility (rolling std of returns)
        returns = close_series.pct_change().dropna()
        if len(returns) < 20:
            return 0.0
        
        short_vol = returns.rolling(10).std().iloc[-1]
        long_vol = returns.rolling(20).std().iloc[-1]
        
        if long_vol > 0:
            vol_ratio = short_vol / long_vol
            # Normalize: >1.5 = high volatility, <0.5 = low volatility
            volatility_regime = (vol_ratio - 1.0) * 2.0
            return np.clip(volatility_regime, -1.0, 1.0)
        return 0.0
    
    def _momentum_regime(self, data: pd.DataFrame, current_idx: int) -> float:
        """Detect momentum regime (1.0 = strong momentum, -1.0 = weak momentum)"""
        close_series = data["Close"] if "Close" in data.columns else data["close"]
        
        if len(close_series) < 14:
            return 0.0
        
        # Use RSI as momentum indicator
        rsi_values = talib.RSI(close_series.values, timeperiod=14)
        if len(rsi_values) > 0 and not np.isnan(rsi_values[-1]):
            # Convert RSI (0-100) to momentum regime (-1 to 1)
            # RSI > 70 = strong upward momentum, RSI < 30 = strong downward momentum
            momentum = (rsi_values[-1] - 50) / 50.0
            return np.clip(momentum, -1.0, 1.0)
        return 0.0
    
    def _volume_regime(self, data: pd.DataFrame, current_idx: int) -> float:
        """Detect volume regime (1.0 = high volume, -1.0 = low volume)"""
        volume_series = data["Volume"] if "Volume" in data.columns else data["volume"]
        
        if len(volume_series) < 20:
            return 0.0
        
        current_volume = volume_series.iloc[-1]
        avg_volume = volume_series.rolling(20).mean().iloc[-1]
        
        if avg_volume > 0:
            volume_ratio = current_volume / avg_volume
            # Normalize: >1.5 = high volume, <0.5 = low volume
            volume_regime = (volume_ratio - 1.0) * 2.0
            return np.clip(volume_regime, -1.0, 1.0)
        return 0.0
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using z-score normalization"""
        # Clip extreme values first
        features = np.clip(features, -10.0, 10.0)
        
        # Z-score normalization
        mean = features.mean()
        std = features.std()
        
        if std > 1e-8:
            features = (features - mean) / std
        else:
            features = features - mean
        
        # Final clip to reasonable range
        features = np.clip(features, -5.0, 5.0)
        
        return features
    
    def extract_features_batch(
        self,
        data: pd.DataFrame,
        indices: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """Extract features for multiple indices (batch processing)
        
        Args:
            data: DataFrame with OHLCV data
            indices: Array of indices to extract features for (if None, uses all valid indices)
        
        Returns:
            Feature matrix (n_samples, 22) or None if insufficient data
        """
        if data is None or len(data) < 50:
            return None
        
        if indices is None:
            # Use all valid indices (starting from index 50)
            indices = np.arange(50, len(data))
        
        features_list = []
        for idx in indices:
            feature = self.extract_features(data, idx)
            if feature is not None:
                features_list.append(feature)
        
        if len(features_list) == 0:
            return None
        
        return np.array(features_list, dtype=np.float32)

