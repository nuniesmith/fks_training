"""
Dual-Head PPO Policy Network

Actor-critic network with shared backbone for trading strategy selection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple
from .networks import BackboneNetwork


class DualHeadPPOPolicy(nn.Module):
    """Dual-head PPO policy with shared backbone for trading model selection"""
    
    def __init__(
        self,
        feature_dim: int = 22,
        hidden_dim: int = 128,
        num_actions: int = 10,  # Number of strategies/models
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Shared backbone for feature extraction
        self.backbone = BackboneNetwork(
            in_features=feature_dim,
            hidden_dimensions=hidden_dim,
            out_features=hidden_dim,
            dropout=dropout
        )
        
        # Actor head (policy) - outputs action logits (before softmax)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_actions)
            # Note: No softmax here - use in forward/get_action for numerical stability
        )
        
        # Critic head (value) - outputs state value
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through network
        
        Args:
            state: Input state tensor (batch_size, feature_dim)
        
        Returns:
            action_logits: Action logits before softmax (batch_size, num_actions)
            value: State value estimate (batch_size, 1)
        """
        # Shared feature extraction
        features = self.backbone(state)
        
        # Actor and critic heads
        action_logits = self.actor(features)
        value = self.critic(features)
        
        return action_logits, value
    
    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[int, float, torch.Tensor, torch.Tensor]:
        """Sample action from policy
        
        Args:
            state: Input state tensor (1, feature_dim) or (feature_dim,)
            deterministic: If True, return most likely action; if False, sample
        
        Returns:
            action: Selected action (int)
            value: State value estimate (float)
            log_prob: Log probability of action (torch.Tensor)
            action_probs: Action probabilities (torch.Tensor)
        """
        # Ensure state is batched
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # Forward pass
        action_logits, value = self.forward(state)
        
        # Apply softmax to get probabilities
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Create distribution
        dist = Categorical(action_probs)
        
        # Sample or take most likely
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        
        return action.item(), value.item(), log_prob, action_probs.squeeze(0)

