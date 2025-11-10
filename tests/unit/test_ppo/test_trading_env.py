"""
Tests for Trading Environment
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, Mock
from src.ppo.trading_env import TradingEnv


class TestTradingEnv:
    """Test TradingEnv"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data"""
        dates = pd.date_range("2020-01-01", periods=200, freq="D")
        data = pd.DataFrame({
            "Open": 100 + np.random.randn(200).cumsum(),
            "High": 105 + np.random.randn(200).cumsum(),
            "Low": 95 + np.random.randn(200).cumsum(),
            "Close": 100 + np.random.randn(200).cumsum(),
            "Volume": 1000000 + np.random.randint(-100000, 100000, 200)
        }, index=dates)
        return data
    
    @patch('src.ppo.trading_env.yf.download')
    def test_initialization_yfinance(self, mock_download, sample_data):
        """Test environment initialization with yfinance"""
        mock_download.return_value = sample_data
        
        env = TradingEnv(
            ticker="AAPL",
            start_date="2020-01-01",
            end_date="2020-12-31",
            data_source="yfinance",
            lookback_window=50
        )
        
        assert env.ticker == "AAPL"
        assert env.observation_space.shape == (22,)  # 22D feature vector
        assert env.action_space.n == 3
    
    def test_reset(self, sample_data):
        """Test environment reset"""
        with patch('src.ppo.trading_env.yf.download', return_value=sample_data):
            env = TradingEnv(
                ticker="AAPL",
                start_date="2020-01-01",
                end_date="2020-12-31",
                data_source="yfinance",
                lookback_window=50
            )
            
            obs, info = env.reset()
            assert obs.shape == env.observation_space.shape
            assert info["balance"] == env.initial_balance
            assert info["shares"] == 0
            assert info["net_worth"] == env.initial_balance
    
    def test_step_hold(self, sample_data):
        """Test step with hold action"""
        with patch('src.ppo.trading_env.yf.download', return_value=sample_data):
            env = TradingEnv(
                ticker="AAPL",
                start_date="2020-01-01",
                end_date="2020-12-31",
                data_source="yfinance",
                lookback_window=50
            )
            
            obs, info = env.reset()
            next_obs, reward, terminated, truncated, next_info = env.step(0)  # Hold
            
            assert next_obs.shape == env.observation_space.shape
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            # Hold should not change shares
            assert next_info["shares"] == info["shares"]
    
    def test_step_buy(self, sample_data):
        """Test step with buy action"""
        with patch('src.ppo.trading_env.yf.download', return_value=sample_data):
            env = TradingEnv(
                ticker="AAPL",
                start_date="2020-01-01",
                end_date="2020-12-31",
                data_source="yfinance",
                lookback_window=50,
                initial_balance=10000.0
            )
            
            obs, info = env.reset()
            next_obs, reward, terminated, truncated, next_info = env.step(1)  # Buy
            
            # Should have bought some shares
            if next_info["shares"] > 0:
                assert next_info["balance"] < info["balance"]
                assert next_info["shares"] > info["shares"]
    
    def test_step_sell(self, sample_data):
        """Test step with sell action"""
        with patch('src.ppo.trading_env.yf.download', return_value=sample_data):
            env = TradingEnv(
                ticker="AAPL",
                start_date="2020-01-01",
                end_date="2020-12-31",
                data_source="yfinance",
                lookback_window=50,
                initial_balance=10000.0
            )
            
            obs, info = env.reset()
            # First buy
            env.step(1)  # Buy
            info_after_buy = env._get_info()
            
            # Then sell
            next_obs, reward, terminated, truncated, next_info = env.step(2)  # Sell
            
            # Should have sold shares
            if info_after_buy["shares"] > 0:
                assert next_info["shares"] == 0
                assert next_info["balance"] > info_after_buy["balance"]
    
    def test_reward_calculation(self, sample_data):
        """Test reward calculation"""
        with patch('src.ppo.trading_env.yf.download', return_value=sample_data):
            env = TradingEnv(
                ticker="AAPL",
                start_date="2020-01-01",
                end_date="2020-12-31",
                data_source="yfinance",
                lookback_window=50
            )
            
            obs, info = env.reset()
            next_obs, reward, terminated, truncated, next_info = env.step(0)  # Hold
            
            # Reward should be in reasonable range (normalized to [-1, 1])
            assert -1.0 <= reward <= 1.0
    
    def test_episode_termination(self, sample_data):
        """Test episode termination"""
        with patch('src.ppo.trading_env.yf.download', return_value=sample_data):
            env = TradingEnv(
                ticker="AAPL",
                start_date="2020-01-01",
                end_date="2020-12-31",
                data_source="yfinance",
                lookback_window=50
            )
            
            obs, info = env.reset()
            
            # Run until termination
            step_count = 0
            done = False
            while not done and step_count < 200:
                obs, reward, terminated, truncated, info = env.step(0)  # Hold
                done = terminated or truncated
                step_count += 1
            
            # Should eventually terminate
            assert done or step_count >= 200

