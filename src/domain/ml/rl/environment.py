"""
Portfolio Management RL Environment

Gym-compatible environment for reinforcement learning-based portfolio allocation.
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple


class PortfolioEnvConfig:
    """Configuration for Portfolio RL Environment"""

    def __init__(
        self,
        initial_balance: float = 10000.0,
        transaction_cost: float = 0.001,  # 0.1% transaction cost
        max_position_size: float = 0.1,  # 10% max position per asset
        rebalancing_frequency: int = 1,  # Rebalance every N steps
        reward_type: str = "sharpe",  # sharpe, returns, or risk_adjusted
        lookback_window: int = 60,  # Number of historical steps to include
    ):
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position_size = max_position_size
        self.rebalancing_frequency = rebalancing_frequency
        self.reward_type = reward_type
        self.lookback_window = lookback_window


class PortfolioEnv(gym.Env):
    """
    Portfolio Management Environment for Reinforcement Learning.

    State Space:
    - Current portfolio weights (n_assets,)
    - Asset prices (n_assets,)
    - Technical indicators (n_assets, n_indicators)
    - Market regime (1,)
    - Portfolio value (1,)

    Action Space:
    - Continuous: Portfolio weights for each asset (n_assets,) summing to 1.0
    - Discrete: Buy/Sell/Hold for each asset (3^n_assets)

    Reward:
    - Sharpe ratio (risk-adjusted returns)
    - Cumulative returns minus transaction costs
    - Risk-adjusted returns
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        price_data: pd.DataFrame,
        config: Optional[PortfolioEnvConfig] = None,
        action_type: str = "continuous",
    ):
        """
        Initialize portfolio environment.

        Args:
            price_data: DataFrame with columns: timestamp, asset_1, asset_2, ...
                       Each column represents price history for an asset
            config: Environment configuration
            action_type: "continuous" for weight allocation, "discrete" for buy/sell/hold
        """
        super().__init__()

        self.config = config or PortfolioEnvConfig()
        self.action_type = action_type
        self.price_data = price_data.copy()

        # Extract asset names (exclude timestamp if present)
        if "timestamp" in self.price_data.columns:
            self.asset_names = [
                col for col in self.price_data.columns if col != "timestamp"
            ]
        else:
            self.asset_names = list(self.price_data.columns)

        self.n_assets = len(self.asset_names)
        self.n_steps = len(self.price_data)

        if self.n_assets == 0:
            raise ValueError("No assets found in price_data")

        # Define action space
        if action_type == "continuous":
            # Continuous: portfolio weights (must sum to 1.0)
            self.action_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.n_assets,),
                dtype=np.float32,
            )
        else:
            # Discrete: 3 actions per asset (buy, sell, hold)
            self.action_space = spaces.MultiDiscrete([3] * self.n_assets)

        # Define observation space
        # State: portfolio weights + prices + technical indicators + portfolio value
        n_price_features = self.n_assets
        n_technical_features = self.n_assets * 3  # RSI, MACD, BB position per asset
        n_state_features = (
            self.n_assets  # Current portfolio weights
            + n_price_features  # Current prices (normalized)
            + n_technical_features  # Technical indicators
            + 1  # Portfolio value (normalized)
            + 1  # Market regime
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_state_features,), dtype=np.float32
        )

        # Initialize state
        self.reset()

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        self.current_step = self.config.lookback_window
        self.portfolio_value = self.config.initial_balance
        self.portfolio_weights = np.ones(self.n_assets) / self.n_assets  # Equal weights
        self.cash = self.config.initial_balance
        self.positions = np.zeros(self.n_assets)  # Number of shares per asset

        # Track portfolio history
        self.portfolio_history = [self.portfolio_value]
        self.returns_history = []

        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: Action to take (portfolio weights or discrete actions)

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Execute action (rebalance portfolio)
        if self.action_type == "continuous":
            # Normalize weights to sum to 1.0
            target_weights = np.clip(action, 0.0, self.config.max_position_size)
            target_weights = target_weights / (target_weights.sum() + 1e-8)
        else:
            # Convert discrete actions to portfolio weights
            target_weights = self._discrete_to_weights(action)

        # Get current prices
        current_prices = self._get_current_prices()

        # Rebalance portfolio
        transaction_cost = self._rebalance_portfolio(target_weights, current_prices)

        # Move to next step
        self.current_step += 1

        # Calculate new portfolio value
        new_prices = self._get_current_prices()
        self.portfolio_value = self._calculate_portfolio_value(new_prices)

        # Calculate reward
        reward = self._calculate_reward(transaction_cost)

        # Update history
        self.portfolio_history.append(self.portfolio_value)
        if len(self.portfolio_history) > 1:
            period_return = (
                self.portfolio_history[-1] - self.portfolio_history[-2]
            ) / self.portfolio_history[-2]
            self.returns_history.append(period_return)

        # Check if done
        terminated = self.current_step >= self.n_steps - 1
        truncated = self.portfolio_value < self.config.initial_balance * 0.1  # 90% loss

        # Get next observation
        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        current_prices = self._get_current_prices()
        normalized_prices = current_prices / current_prices.mean()

        # Get technical indicators (simplified - would use actual indicators in production)
        technical_features = self._get_technical_features()

        # Portfolio value (normalized)
        normalized_portfolio_value = (
            self.portfolio_value / self.config.initial_balance
        )

        # Market regime (simplified - 0=calm, 1=transition, 2=volatile)
        market_regime = self._get_market_regime()

        # Combine all features
        observation = np.concatenate(
            [
                self.portfolio_weights,  # Current portfolio weights
                normalized_prices,  # Normalized current prices
                technical_features.flatten(),  # Technical indicators
                [normalized_portfolio_value],  # Portfolio value
                [market_regime],  # Market regime
            ]
        )

        return observation.astype(np.float32)

    def _get_current_prices(self) -> np.ndarray:
        """Get current asset prices."""
        if self.current_step >= len(self.price_data):
            # Use last available price
            return self.price_data[self.asset_names].iloc[-1].values

        return self.price_data[self.asset_names].iloc[self.current_step].values

    def _get_technical_features(self) -> np.ndarray:
        """Get technical indicators for current step."""
        # Simplified: return RSI, MACD signal, BB position for each asset
        # In production, would calculate actual indicators
        n_features = 3  # RSI, MACD, BB
        features = np.random.rand(self.n_assets, n_features)  # Placeholder

        # Calculate actual indicators if we have enough history
        if self.current_step >= 20:
            window_data = self.price_data[self.asset_names].iloc[
                max(0, self.current_step - 20) : self.current_step + 1
            ]

            for i, asset in enumerate(self.asset_names):
                prices = window_data[asset].values

                # Simplified RSI
                if len(prices) >= 14:
                    delta = np.diff(prices)
                    gains = np.where(delta > 0, delta, 0)
                    losses = np.where(delta < 0, -delta, 0)
                    avg_gain = np.mean(gains[-14:])
                    avg_loss = np.mean(losses[-14:])
                    if avg_loss > 0:
                        rs = avg_gain / avg_loss
                        features[i, 0] = 100 - (100 / (1 + rs)) / 100  # Normalized
                    else:
                        features[i, 0] = 0.5

                # Simplified MACD signal
                if len(prices) >= 26:
                    ema_12 = np.mean(prices[-12:])
                    ema_26 = np.mean(prices[-26:])
                    macd = (ema_12 - ema_26) / prices[-1]  # Normalized
                    features[i, 1] = macd

                # Simplified BB position
                if len(prices) >= 20:
                    sma = np.mean(prices[-20:])
                    std = np.std(prices[-20:])
                    if std > 0:
                        bb_position = (prices[-1] - (sma - 2 * std)) / (4 * std)
                        features[i, 2] = np.clip(bb_position, 0, 1)
                    else:
                        features[i, 2] = 0.5

        return features

    def _get_market_regime(self) -> float:
        """Get current market regime (0=calm, 1=transition, 2=volatile)."""
        if len(self.returns_history) < 20:
            return 1.0  # Transition

        recent_volatility = np.std(self.returns_history[-20:])
        if recent_volatility < 0.01:
            return 0.0  # Calm
        elif recent_volatility > 0.03:
            return 2.0  # Volatile
        else:
            return 1.0  # Transition

    def _rebalance_portfolio(
        self, target_weights: np.ndarray, current_prices: np.ndarray
    ) -> float:
        """
        Rebalance portfolio to target weights.

        Returns:
            Total transaction cost
        """
        # Calculate target portfolio value in each asset
        target_values = target_weights * self.portfolio_value

        # Calculate current portfolio value in each asset
        current_values = self.portfolio_weights * self.portfolio_value

        # Calculate trades needed
        trades = target_values - current_values
        total_trade_value = np.abs(trades).sum()

        # Calculate transaction cost
        transaction_cost = total_trade_value * self.config.transaction_cost

        # Update portfolio weights
        self.portfolio_weights = target_weights

        # Update cash (simplified - assumes we can always rebalance)
        self.cash = self.portfolio_value - total_trade_value - transaction_cost

        return transaction_cost

    def _calculate_portfolio_value(self, prices: np.ndarray) -> float:
        """Calculate current portfolio value."""
        # Simplified: portfolio value based on weights
        # In production, would track actual positions
        return self.portfolio_value  # Simplified - assumes prices don't change within step

    def _calculate_reward(self, transaction_cost: float) -> float:
        """Calculate reward based on portfolio performance."""
        if len(self.returns_history) == 0:
            return 0.0

        if self.config.reward_type == "sharpe":
            # Sharpe ratio
            if len(self.returns_history) < 2:
                return 0.0

            mean_return = np.mean(self.returns_history)
            std_return = np.std(self.returns_history)

            if std_return > 0:
                sharpe = mean_return / std_return * np.sqrt(252)  # Annualized
                return sharpe
            else:
                return 0.0

        elif self.config.reward_type == "returns":
            # Cumulative returns minus transaction costs
            total_return = (self.portfolio_value - self.config.initial_balance) / self.config.initial_balance
            return total_return - (transaction_cost / self.config.initial_balance)

        else:  # risk_adjusted
            # Returns divided by volatility
            if len(self.returns_history) < 2:
                return 0.0

            mean_return = np.mean(self.returns_history)
            std_return = np.std(self.returns_history)

            if std_return > 0:
                return mean_return / std_return
            else:
                return mean_return

    def _discrete_to_weights(self, actions: np.ndarray) -> np.ndarray:
        """Convert discrete actions to portfolio weights."""
        # 0 = sell, 1 = hold, 2 = buy
        weights = np.ones(self.n_assets) / self.n_assets  # Start with equal weights

        for i, action in enumerate(actions):
            if action == 0:  # Sell
                weights[i] *= 0.5
            elif action == 2:  # Buy
                weights[i] *= 1.5

        # Normalize
        weights = weights / weights.sum()
        return weights

    def _get_info(self) -> Dict:
        """Get additional info about current state."""
        return {
            "portfolio_value": float(self.portfolio_value),
            "portfolio_weights": self.portfolio_weights.tolist(),
            "step": int(self.current_step),
            "total_return": float(
                (self.portfolio_value - self.config.initial_balance)
                / self.config.initial_balance
            ),
        }

    def render(self, mode: str = "human"):
        """Render environment (optional)."""
        if mode == "human":
            print(f"Step: {self.current_step}")
            print(f"Portfolio Value: ${self.portfolio_value:.2f}")
            print(f"Portfolio Weights: {self.portfolio_weights}")
            print(f"Total Return: {(self.portfolio_value - self.config.initial_balance) / self.config.initial_balance * 100:.2f}%")

