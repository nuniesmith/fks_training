"""
Technical Indicators for Feature Engineering

Implements common technical indicators used in trading:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- SMA/EMA (Simple/Exponential Moving Averages)
- And more...
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


class TechnicalIndicators:
    """
    Collection of technical indicators for feature engineering.

    All methods accept pandas DataFrames with OHLCV columns:
    - open, high, low, close, volume
    """

    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).

        Args:
            prices: Series of closing prices
            period: RSI period (default: 14)

        Returns:
            Series of RSI values (0-100)
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Fill NaN with neutral value

    @staticmethod
    def macd(
        prices: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        Args:
            prices: Series of closing prices
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            signal_period: Signal line EMA period (default: 9)

        Returns:
            Dictionary with 'macd', 'signal', and 'histogram' series
        """
        ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
        ema_slow = prices.ewm(span=slow_period, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        return {
            "macd": macd_line,
            "signal": signal_line,
            "histogram": histogram,
        }

    @staticmethod
    def bollinger_bands(
        prices: pd.Series, period: int = 20, num_std: float = 2.0
    ) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands.

        Args:
            prices: Series of closing prices
            period: Moving average period (default: 20)
            num_std: Number of standard deviations (default: 2.0)

        Returns:
            Dictionary with 'upper', 'middle', and 'lower' band series
        """
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()

        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)

        return {
            "upper": upper_band,
            "middle": sma,
            "lower": lower_band,
        }

    @staticmethod
    def sma(prices: pd.Series, period: int) -> pd.Series:
        """
        Calculate Simple Moving Average (SMA).

        Args:
            prices: Series of prices
            period: Moving average period

        Returns:
            Series of SMA values
        """
        return prices.rolling(window=period).mean()

    @staticmethod
    def ema(prices: pd.Series, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average (EMA).

        Args:
            prices: Series of prices
            period: EMA period

        Returns:
            Series of EMA values
        """
        return prices.ewm(span=period, adjust=False).mean()

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).

        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            period: ATR period (default: 14)

        Returns:
            Series of ATR values
        """
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()

        return atr

    @staticmethod
    def stochastic(
        high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3
    ) -> Dict[str, pd.Series]:
        """
        Calculate Stochastic Oscillator.

        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            k_period: %K period (default: 14)
            d_period: %D period (default: 3)

        Returns:
            Dictionary with 'k' and 'd' series
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d = k.rolling(window=d_period).mean()

        return {"k": k, "d": d}

    @staticmethod
    def adx(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """
        Calculate Average Directional Index (ADX).

        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            period: ADX period (default: 14)

        Returns:
            Series of ADX values (0-100)
        """
        # Calculate +DM and -DM
        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        # Calculate True Range
        atr = TechnicalIndicators.atr(high, low, close, period)

        # Calculate +DI and -DI
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        # Calculate DX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)

        # Calculate ADX
        adx = dx.rolling(window=period).mean()

        return adx.fillna(0)

    @staticmethod
    def volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculate Volume Simple Moving Average.

        Args:
            volume: Series of volume values
            period: Moving average period (default: 20)

        Returns:
            Series of volume SMA values
        """
        return volume.rolling(window=period).mean()

    @staticmethod
    def create_all_indicators(
        df: pd.DataFrame,
        include_rsi: bool = True,
        include_macd: bool = True,
        include_bollinger: bool = True,
        include_sma: bool = True,
        include_ema: bool = True,
        include_atr: bool = True,
        include_stochastic: bool = False,
        include_adx: bool = False,
    ) -> pd.DataFrame:
        """
        Create all technical indicators for a DataFrame with OHLCV data.

        Args:
            df: DataFrame with columns: open, high, low, close, volume
            include_rsi: Include RSI indicator
            include_macd: Include MACD indicator
            include_bollinger: Include Bollinger Bands
            include_sma: Include SMA indicators
            include_ema: Include EMA indicators
            include_atr: Include ATR indicator
            include_stochastic: Include Stochastic Oscillator
            include_adx: Include ADX indicator

        Returns:
            DataFrame with original columns plus indicator columns
        """
        result_df = df.copy()

        if "close" not in result_df.columns:
            raise ValueError("DataFrame must contain 'close' column")

        # RSI
        if include_rsi:
            result_df["rsi_14"] = TechnicalIndicators.rsi(result_df["close"], period=14)

        # MACD
        if include_macd:
            macd_data = TechnicalIndicators.macd(result_df["close"])
            result_df["macd"] = macd_data["macd"]
            result_df["macd_signal"] = macd_data["signal"]
            result_df["macd_histogram"] = macd_data["histogram"]

        # Bollinger Bands
        if include_bollinger:
            bb_data = TechnicalIndicators.bollinger_bands(result_df["close"])
            result_df["bb_upper"] = bb_data["upper"]
            result_df["bb_middle"] = bb_data["middle"]
            result_df["bb_lower"] = bb_data["lower"]
            # Also add distance from bands as features
            result_df["bb_width"] = (bb_data["upper"] - bb_data["lower"]) / bb_data["middle"]
            result_df["bb_position"] = (
                (result_df["close"] - bb_data["lower"]) / (bb_data["upper"] - bb_data["lower"])
            )

        # SMA
        if include_sma:
            result_df["sma_20"] = TechnicalIndicators.sma(result_df["close"], period=20)
            result_df["sma_50"] = TechnicalIndicators.sma(result_df["close"], period=50)
            result_df["sma_200"] = TechnicalIndicators.sma(result_df["close"], period=200)

        # EMA
        if include_ema:
            result_df["ema_12"] = TechnicalIndicators.ema(result_df["close"], period=12)
            result_df["ema_26"] = TechnicalIndicators.ema(result_df["close"], period=26)

        # ATR
        if include_atr and all(col in result_df.columns for col in ["high", "low", "close"]):
            result_df["atr_14"] = TechnicalIndicators.atr(
                result_df["high"], result_df["low"], result_df["close"], period=14
            )

        # Stochastic
        if include_stochastic and all(
            col in result_df.columns for col in ["high", "low", "close"]
        ):
            stoch_data = TechnicalIndicators.stochastic(
                result_df["high"], result_df["low"], result_df["close"]
            )
            result_df["stoch_k"] = stoch_data["k"]
            result_df["stoch_d"] = stoch_data["d"]

        # ADX
        if include_adx and all(col in result_df.columns for col in ["high", "low", "close"]):
            result_df["adx_14"] = TechnicalIndicators.adx(
                result_df["high"], result_df["low"], result_df["close"], period=14
            )

        # Volume indicators
        if "volume" in result_df.columns:
            result_df["volume_sma_20"] = TechnicalIndicators.volume_sma(
                result_df["volume"], period=20
            )
            result_df["volume_ratio"] = result_df["volume"] / result_df["volume_sma_20"]

        return result_df

