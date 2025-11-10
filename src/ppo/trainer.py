"""
PPO Trainer with Clipped Surrogate Objective

Implements PPO algorithm with clipped policy updates for stable training.
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions import Categorical
from typing import Dict
from loguru import logger
from .policy_network import DualHeadPPOPolicy


class PPOTrainer:
    """PPO trainer with clipped surrogate objective for meta-learning"""
    
    def __init__(
        self,
        policy: DualHeadPPOPolicy,
        lr: float = 0.001,  # Learning rate (from research: 0.001 works well)
        gamma: float = 0.99,  # Discount factor
        epsilon: float = 0.2,  # Clipping parameter (from research: 0.2 is standard)
        value_coef: float = 0.5,  # Value loss coefficient
        entropy_coef: float = 0.01,  # Entropy coefficient (encourages exploration)
        max_grad_norm: float = 0.5,  # Gradient clipping
        ppo_epochs: int = 10,  # Number of PPO update epochs
        batch_size: int = 128  # Batch size for updates
    ):
        self.policy = policy
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.gamma = gamma
        self.epsilon = epsilon  # Clip range [1-ε, 1+ε]
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
    
    def update_policy(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor
    ) -> Dict[str, float]:
        """Update policy using PPO with clipped surrogate objective
        
        PPO Algorithm:
        1. Collect trajectories under current policy
        2. Compute advantages using GAE
        3. For K epochs:
           a. Create mini-batches
           b. Compute policy ratio: r(θ) = π_θ(a|s) / π_θ_old(a|s)
           c. Compute clipped surrogate: L^CLIP = min(r * A, clip(r, 1-ε, 1+ε) * A)
           d. Compute value loss: L^VF = (V(s) - R)^2
           e. Compute entropy bonus: L^S = -H(π(·|s))
           f. Total loss: L = -L^CLIP + c_vf * L^VF - c_s * L^S
           g. Update policy with gradient descent
        
        Args:
            states: Collected states (n_steps, feature_dim)
            actions: Collected actions (n_steps,)
            old_log_probs: Old log probabilities (n_steps,)
            advantages: Computed advantages (n_steps,)
            returns: Computed returns (n_steps,)
        
        Returns:
            Dictionary with loss statistics
        """
        # Handle empty batches
        if len(states) == 0:
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "total_loss": 0.0,
                "n_updates": 0
            }
        
        # Create dataset for mini-batching
        dataset = TensorDataset(states, actions, old_log_probs, advantages, returns)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        total_pol_loss = 0.0
        total_val_loss = 0.0
        total_entropy = 0.0
        total_loss = 0.0
        n_updates = 0
        
        # Multiple epochs of updates on same data (PPO key feature)
        for epoch in range(self.ppo_epochs):
            for batch in loader:
                s, a, old_lp, adv_batch, ret = batch
                
                # Forward pass through policy
                action_logits, v_pred = self.policy(s)
                action_probs = F.softmax(action_logits, dim=-1)
                dist = Categorical(action_probs)
                
                # Compute new log probabilities
                new_logp = dist.log_prob(a)
                
                # Compute entropy (for exploration bonus)
                entropy = dist.entropy().mean()
                
                # Compute ratio: r(θ) = π_θ(a|s) / π_θ_old(a|s)
                # In log space: ratio = exp(new_logp - old_logp)
                ratio = torch.exp(new_logp - old_lp.detach())
                
                # Clipped surrogate objective
                # L^CLIP(θ) = E[min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)]
                surr1 = ratio * adv_batch.detach()
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * adv_batch.detach()
                pol_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (squared error between predicted and actual returns)
                # Use smooth L1 loss for robustness (from research)
                val_loss = F.smooth_l1_loss(v_pred.squeeze(-1), ret.detach()).mean()
                
                # Total loss
                # L = -L^CLIP + c_vf * L^VF - c_s * L^S
                loss = pol_loss + self.value_coef * val_loss - self.entropy_coef * entropy
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping (prevents exploding gradients)
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                
                # Track statistics
                total_pol_loss += pol_loss.item()
                total_val_loss += val_loss.item()
                total_entropy += entropy.item()
                total_loss += loss.item()
                n_updates += 1
        
        # Average statistics
        if n_updates > 0:
            avg_pol_loss = total_pol_loss / n_updates
            avg_val_loss = total_val_loss / n_updates
            avg_entropy = total_entropy / n_updates
            avg_loss = total_loss / n_updates
        else:
            avg_pol_loss = avg_val_loss = avg_entropy = avg_loss = 0.0
        
        logger.info(
            f"PPO update complete: "
            f"pol_loss={avg_pol_loss:.4f}, "
            f"val_loss={avg_val_loss:.4f}, "
            f"entropy={avg_entropy:.4f}, "
            f"total_loss={avg_loss:.4f}"
        )
        
        return {
            "policy_loss": avg_pol_loss,
            "value_loss": avg_val_loss,
            "entropy": avg_entropy,
            "total_loss": avg_loss,
            "n_updates": n_updates
        }

