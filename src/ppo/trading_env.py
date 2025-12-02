"""
Trading Environment for PPO Training

Gymnasium-compatible environment for single-asset trading (buy/sell/hold).
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from loguru import logger
import yfinance as yf
from ta import add_all_ta_features
import httpx
from .feature_extractor import FKSFeatureExtractor


class TradingEnv(gym.Env):
    """Gymnasium-compatible trading environment for PPO training
    
    State: OHLCV data + technical indicators
    Actions: 0 (hold), 1 (buy), 2 (sell)
    Reward: Profit minus transaction costs, normalized to [-1, 1]
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        ticker: str = "AAPL",
        start_date: str = "2020-01-01",
        end_date: str = "2025-11-01",
        initial_balance: float = 10000.0,
        transaction_cost: float = 0.001,  # 0.1% transaction cost
        slippage: float = 0.0005,  # 0.05% slippage
        data_source: str = "yfinance",  # "yfinance" or "fks_data"
        data_service_url: str = "http://fks_data:8003",
        normalize_states: bool = True,
        lookback_window: int = 50
    ):
        """Initialize trading environment
        
        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")
            start_date: Start date for historical data
            end_date: End date for historical data
            initial_balance: Starting cash balance
            transaction_cost: Transaction cost as fraction (0.001 = 0.1%)
            slippage: Slippage as fraction (0.0005 = 0.05%)
            data_source: Data source ("yfinance" or "fks_data")
            data_service_url: URL for fks_data service
            normalize_states: Whether to normalize state features
            lookback_window: Number of historical bars to include in state
        """
        super().__init__()
        
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.data_source = data_source
        self.data_service_url = data_service_url
        self.normalize_states = normalize_states
        self.lookback_window = lookback_window
        
        # Load and prepare data
        self.data = self._load_data()
        
        if self.data is None or len(self.data) < self.lookback_window + 10:
            raise ValueError(f"Insufficient data for {ticker}")
        
        # Add technical indicators
        self.data = self._add_indicators(self.data)
        
        # Drop NaN rows (from indicator calculations)
        self.data = self.data.dropna()
        
        if len(self.data) < self.lookback_window + 10:
            raise ValueError(f"Insufficient data after adding indicators for {ticker}")
        
        # Initialize feature extractor (22D feature vector)
        self.feature_extractor = FKSFeatureExtractor(normalize=normalize_states)
        
        # Define observation space (22D feature vector)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(22,),  # 22D feature vector
            dtype=np.float32
        )
        
        # Define action space (0: hold, 1: buy, 2: sell)
        self.action_space = gym.spaces.Discrete(3)
        
        # Initialize state
        self.reset()
    
    def _load_data(self) -> Optional[pd.DataFrame]:
        """Load historical data from yfinance or fks_data"""
        if self.data_source == "yfinance":
            logger.info(f"Loading data from yfinance for {self.ticker}...")
            try:
                data = yf.download(
                    self.ticker,
                    start=self.start_date,
                    end=self.end_date,
                    progress=False
                )
                if data.empty:
                    logger.error(f"No data retrieved for {self.ticker}")
                    return None
                return data
            except Exception as e:
                logger.error(f"Failed to load data from yfinance: {e}")
                return None
        elif self.data_source == "fks_data":
            logger.info(f"Loading data from fks_data for {self.ticker}...")
            try:
                # Fetch from fks_data service (synchronous for simplicity)
                response = httpx.get(
                    f"{self.data_service_url}/api/v1/data/{self.ticker}",
                    params={
                        "start_date": self.start_date,
                        "end_date": self.end_date,
                        "interval": "1d"
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                data_dict = response.json()
                
                # Convert to DataFrame
                data = pd.DataFrame(data_dict.get("data", []))
                if data.empty:
                    logger.error(f"No data retrieved from fks_data for {self.ticker}")
                    return None
                
                # Rename columns to match yfinance format
                if "timestamp" in data.columns:
                    data["Date"] = pd.to_datetime(data["timestamp"])
                    data.set_index("Date", inplace=True)
                if "open" in data.columns:
                    data.rename(columns={
                        "open": "Open",
                        "high": "High",
                        "low": "Low",
                        "close": "Close",
                        "volume": "Volume"
                    }, inplace=True)
                
                return data[["Open", "High", "Low", "Close", "Volume"]]
            except Exception as e:
                logger.error(f"Failed to load data from fks_data: {e}")
                return None
        else:
            raise ValueError(f"Unknown data source: {self.data_source}")
    
    def _add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators using ta library"""
        try:
            # Add all technical indicators
            data = add_all_ta_features(
                data,
                open="Open",
                high="High",
                low="Low",
                close="Close",
                volume="Volume"
            )
            logger.info(f"Added technical indicators. Shape: {data.shape}")
            return data
        except Exception as e:
            logger.warning(f"Failed to add some indicators: {e}")
            # Fallback: add basic indicators manually
            data["sma_20"] = data["Close"].rolling(20).mean()
            data["sma_50"] = data["Close"].rolling(50).mean()
            data["rsi"] = self._calculate_rsi(data["Close"], 14)
            return data
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Start after indicator warm-up period
        self.current_step = self.lookback_window
        
        # Reset trading state
        self.balance = self.initial_balance
        self.shares = 0
        self.net_worth = self.balance
        self.prev_worth = self.initial_balance
        
        # Trading history
        self.trades = []
        self.portfolio_history = [self.net_worth]
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment
        
        Args:
            action: 0 (hold), 1 (buy), 2 (sell)
        
        Returns:
            observation: Next state observation
            reward: Reward for this step
            terminated: Whether episode is done
            truncated: Whether episode was truncated
            info: Additional information
        """
        # Get current price
        current_price = self.data["Close"].iloc[self.current_step]
        
        # Store previous net worth
        self.prev_worth = self.net_worth
        
        # Execute action
        if action == 1:  # Buy
            self._execute_buy(current_price)
        elif action == 2:  # Sell
            self._execute_sell(current_price)
        # action == 0: Hold (do nothing)
        
        # Update net worth
        self.net_worth = self.balance + self.shares * current_price
        
        # Calculate reward (normalized profit)
        reward = self._calculate_reward()
        
        # Move to next step
        self.current_step += 1
        
        # Update history
        self.portfolio_history.append(self.net_worth)
        
        # Check if done
        terminated = self.current_step >= len(self.data) - 1
        truncated = self.net_worth < self.initial_balance * 0.1  # 90% loss
        
        # Get next observation
        observation = self._get_observation() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _execute_buy(self, price: float):
        """Execute buy action"""
        if self.balance > price:
            # Calculate number of shares we can buy (with fees)
            cost_per_share = price * (1 + self.transaction_cost + self.slippage)
            shares_to_buy = int(self.balance / cost_per_share)
            
            if shares_to_buy > 0:
                cost = shares_to_buy * cost_per_share
                self.shares += shares_to_buy
                self.balance -= cost
                self.trades.append({
                    "step": self.current_step,
                    "action": "buy",
                    "price": price,
                    "shares": shares_to_buy,
                    "cost": cost
                })
    
    def _execute_sell(self, price: float):
        """Execute sell action"""
        if self.shares > 0:
            # Calculate revenue (with fees)
            revenue_per_share = price * (1 - self.transaction_cost - self.slippage)
            revenue = self.shares * revenue_per_share
            
            self.trades.append({
                "step": self.current_step,
                "action": "sell",
                "price": price,
                "shares": self.shares,
                "revenue": revenue
            })
            
            self.balance += revenue
            self.shares = 0
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on profit
        
        Reward is normalized to [-1, 1] for stability:
        - Positive reward for profit
        - Negative reward for loss
        - Clipped to prevent extreme values
        """
        # Calculate profit/loss
        profit = self.net_worth - self.prev_worth
        profit_pct = profit / self.prev_worth if self.prev_worth > 0 else 0.0
        
        # Normalize reward (clip to [-1, 1])
        reward = np.clip(profit_pct * 10, -1.0, 1.0)
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation (22D feature vector)"""
        if self.current_step >= len(self.data):
            return np.zeros(22, dtype=np.float32)
        
        # Extract 22D feature vector using feature extractor
        feature_vector = self.feature_extractor.extract_features(
            self.data,
            current_idx=self.current_step
        )
        
        if feature_vector is None:
            # Fallback: return zeros if feature extraction fails
            logger.warning(f"Feature extraction failed at step {self.current_step}")
            return np.zeros(22, dtype=np.float32)
        
        return feature_vector
    
    def _get_info(self) -> Dict:
        """Get additional information about current state"""
        return {
            "balance": self.balance,
            "shares": self.shares,
            "net_worth": self.net_worth,
            "current_step": self.current_step,
            "total_trades": len(self.trades),
            "profit": self.net_worth - self.initial_balance,
            "profit_pct": (self.net_worth - self.initial_balance) / self.initial_balance * 100
        }
    
    def render(self, mode: str = "human"):
        """Render environment (for visualization)"""
        if mode == "human":
            print(f"Step: {self.current_step}")
            print(f"Balance: ${self.balance:.2f}")
            print(f"Shares: {self.shares}")
            print(f"Net Worth: ${self.net_worth:.2f}")
            print(f"Profit: ${self.net_worth - self.initial_balance:.2f}")

