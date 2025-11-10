"""
Tests for PPO Trainer
"""
import pytest
import torch
from src.ppo.trainer import PPOTrainer
from src.ppo.policy_network import DualHeadPPOPolicy


class TestPPOTrainer:
    """Test PPOTrainer"""
    
    def test_initialization(self):
        """Test trainer initialization"""
        policy = DualHeadPPOPolicy(feature_dim=22, hidden_dim=128, num_actions=10)
        trainer = PPOTrainer(policy=policy)
        assert trainer is not None
        assert trainer.policy == policy
    
    def test_update_policy(self):
        """Test policy update"""
        policy = DualHeadPPOPolicy(feature_dim=22, hidden_dim=128, num_actions=10)
        trainer = PPOTrainer(
            policy=policy,
            lr=0.001,
            epsilon=0.2,
            ppo_epochs=2,
            batch_size=32
        )
        
        # Create dummy trajectory data
        n_steps = 100
        states = torch.randn(n_steps, 22)
        actions = torch.randint(0, 10, (n_steps,))
        old_log_probs = torch.randn(n_steps)
        advantages = torch.randn(n_steps)
        returns = torch.randn(n_steps)
        
        # Update policy
        stats = trainer.update_policy(
            states=states,
            actions=actions,
            old_log_probs=old_log_probs,
            advantages=advantages,
            returns=returns
        )
        
        assert "policy_loss" in stats
        assert "value_loss" in stats
        assert "entropy" in stats
        assert "total_loss" in stats
        assert "n_updates" in stats
        assert stats["n_updates"] > 0
    
    def test_update_policy_empty_batch(self):
        """Test update with empty batch"""
        policy = DualHeadPPOPolicy(feature_dim=22, hidden_dim=128, num_actions=10)
        trainer = PPOTrainer(policy=policy)
        
        # Empty batch
        states = torch.tensor([], dtype=torch.float32).reshape(0, 22)
        actions = torch.tensor([], dtype=torch.long)
        old_log_probs = torch.tensor([], dtype=torch.float32)
        advantages = torch.tensor([], dtype=torch.float32)
        returns = torch.tensor([], dtype=torch.float32)
        
        stats = trainer.update_policy(
            states=states,
            actions=actions,
            old_log_probs=old_log_probs,
            advantages=advantages,
            returns=returns
        )
        
        assert stats["n_updates"] == 0
        assert stats["policy_loss"] == 0.0
    
    def test_update_policy_small_batch(self):
        """Test update with small batch"""
        policy = DualHeadPPOPolicy(feature_dim=22, hidden_dim=128, num_actions=10)
        trainer = PPOTrainer(
            policy=policy,
            batch_size=10,
            ppo_epochs=1
        )
        
        # Small batch
        n_steps = 5
        states = torch.randn(n_steps, 22)
        actions = torch.randint(0, 10, (n_steps,))
        old_log_probs = torch.randn(n_steps)
        advantages = torch.randn(n_steps)
        returns = torch.randn(n_steps)
        
        stats = trainer.update_policy(
            states=states,
            actions=actions,
            old_log_probs=old_log_probs,
            advantages=advantages,
            returns=returns
        )
        
        assert stats["n_updates"] > 0

