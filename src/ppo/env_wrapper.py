"""
Environment Wrapper for Vectorized Trading Environments

Utility for creating vectorized environments for parallel training.
"""

import numpy as np
from typing import List
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from .trading_env import TradingEnv


def make_trading_env(
    ticker: str,
    start_date: str,
    end_date: str,
    n_envs: int = 1,
    **env_kwargs
):
    """Create vectorized trading environment
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date for data
        end_date: End date for data
        n_envs: Number of parallel environments
        **env_kwargs: Additional environment arguments
    
    Returns:
        Vectorized environment (SubprocVecEnv or DummyVecEnv)
    """
    def make_env():
        return TradingEnv(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            **env_kwargs
        )
    
    if n_envs == 1:
        return DummyVecEnv([make_env])
    else:
        return SubprocVecEnv([make_env for _ in range(n_envs)])

