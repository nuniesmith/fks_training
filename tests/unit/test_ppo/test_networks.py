"""
Tests for PPO Networks
"""
import pytest
import torch
from src.ppo.networks import BackboneNetwork
from src.ppo.policy_network import DualHeadPPOPolicy


class TestBackboneNetwork:
    """Test BackboneNetwork"""
    
    def test_initialization(self):
        """Test network initialization"""
        network = BackboneNetwork(
            in_features=22,
            hidden_dimensions=128,
            out_features=64,
            dropout=0.2
        )
        assert network is not None
    
    def test_forward_pass(self):
        """Test forward pass"""
        network = BackboneNetwork(in_features=22, hidden_dimensions=128, out_features=64)
        x = torch.randn(32, 22)  # Batch size 32, 22 features
        output = network(x)
        assert output.shape == (32, 64)
    
    def test_different_input_sizes(self):
        """Test with different input sizes"""
        network = BackboneNetwork(in_features=10, hidden_dimensions=64, out_features=32)
        x = torch.randn(1, 10)
        output = network(x)
        assert output.shape == (1, 32)


class TestDualHeadPPOPolicy:
    """Test DualHeadPPOPolicy"""
    
    def test_initialization(self):
        """Test policy initialization"""
        policy = DualHeadPPOPolicy(
            feature_dim=22,
            hidden_dim=128,
            num_actions=10,
            dropout=0.2
        )
        assert policy is not None
    
    def test_forward_pass(self):
        """Test forward pass"""
        policy = DualHeadPPOPolicy(feature_dim=22, hidden_dim=128, num_actions=10)
        state = torch.randn(32, 22)  # Batch size 32
        action_logits, value = policy(state)
        assert action_logits.shape == (32, 10)  # 10 actions
        assert value.shape == (32, 1)  # Value
    
    def test_get_action(self):
        """Test action sampling"""
        policy = DualHeadPPOPolicy(feature_dim=22, hidden_dim=128, num_actions=10)
        state = torch.randn(22)  # Single state
        action, value, log_prob, action_probs = policy.get_action(state, deterministic=False)
        assert isinstance(action, int)
        assert 0 <= action < 10
        assert isinstance(value, float)
        assert log_prob.shape == ()
        assert action_probs.shape == (10,)
        assert torch.allclose(action_probs.sum(), torch.tensor(1.0), atol=1e-5)
    
    def test_get_action_deterministic(self):
        """Test deterministic action"""
        policy = DualHeadPPOPolicy(feature_dim=22, hidden_dim=128, num_actions=10)
        state = torch.randn(22)
        action, value, log_prob, action_probs = policy.get_action(state, deterministic=True)
        assert isinstance(action, int)
        assert 0 <= action < 10
        # Deterministic should return most likely action
        expected_action = torch.argmax(action_probs).item()
        assert action == expected_action
    
    def test_batch_get_action(self):
        """Test action sampling with batched states"""
        policy = DualHeadPPOPolicy(feature_dim=22, hidden_dim=128, num_actions=10)
        state = torch.randn(1, 22)  # Batched state
        action, value, log_prob, action_probs = policy.get_action(state, deterministic=False)
        assert isinstance(action, int)
        assert 0 <= action < 10

