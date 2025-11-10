"""
Tests for PPO Data Collection
"""
import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
from src.ppo.data_collection import compute_returns, compute_gae_advantages
from src.ppo.policy_network import DualHeadPPOPolicy


class TestComputeReturns:
    """Test return computation"""
    
    def test_compute_returns(self):
        """Test discounted returns"""
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        gamma = 0.9
        returns = compute_returns(rewards, gamma)
        
        # Manual calculation:
        # G_3 = 4.0
        # G_2 = 3.0 + 0.9 * 4.0 = 6.6
        # G_1 = 2.0 + 0.9 * 6.6 = 7.94
        # G_0 = 1.0 + 0.9 * 7.94 = 8.146
        assert len(returns) == len(rewards)
        assert returns[-1].item() == pytest.approx(4.0, abs=0.01)
        assert returns[0].item() > returns[-1].item()  # Earlier returns should be higher
    
    def test_compute_returns_empty(self):
        """Test with empty rewards"""
        rewards = torch.tensor([], dtype=torch.float32)
        returns = compute_returns(rewards, 0.9)
        assert len(returns) == 0
    
    def test_compute_returns_gamma_one(self):
        """Test with gamma=1.0 (no discounting)"""
        rewards = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        returns = compute_returns(rewards, 1.0)
        # With gamma=1, returns should be cumulative sum in reverse
        assert returns[-1].item() == pytest.approx(3.0, abs=0.01)
        assert returns[0].item() == pytest.approx(6.0, abs=0.01)


class TestComputeGAE:
    """Test GAE advantage computation"""
    
    def test_compute_gae_advantages(self):
        """Test GAE advantages"""
        rewards = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        values = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float32)
        gamma = 0.9
        lambda_gae = 0.95
        
        advantages = compute_gae_advantages(rewards, values, gamma, lambda_gae)
        
        assert len(advantages) == len(rewards)
        # Advantages should have similar magnitude to rewards
        assert advantages.abs().max() > 0
    
    def test_compute_gae_empty(self):
        """Test with empty rewards"""
        rewards = torch.tensor([], dtype=torch.float32)
        values = torch.tensor([], dtype=torch.float32)
        advantages = compute_gae_advantages(rewards, values, 0.9, 0.95)
        assert len(advantages) == 0
    
    def test_compute_gae_positive_rewards(self):
        """Test with positive rewards"""
        rewards = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        values = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
        advantages = compute_gae_advantages(rewards, values, 0.9, 0.95)
        # With positive rewards and low values, advantages should be positive
        assert advantages.mean().item() > 0

