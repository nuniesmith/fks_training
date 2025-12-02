"""
Data Collection for PPO Training

Collects trajectories from environment and computes returns/advantages using GAE.
"""

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple
import numpy as np
from loguru import logger
from .policy_network import DualHeadPPOPolicy


def forward_pass(
    env,
    agent: DualHeadPPOPolicy,
    gamma: float = 0.99,
    max_steps: int = 1000
) -> Tuple[float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect trajectory data from environment
    
    Args:
        env: Trading environment (Gymnasium-compatible)
        agent: PPO policy network
        gamma: Discount factor
        max_steps: Maximum steps per episode
    
    Returns:
        total_reward: Sum of rewards in episode
        states: Collected states (n_steps, feature_dim)
        actions: Collected actions (n_steps,)
        old_log_probs: Log probabilities of actions (n_steps,)
        advantages: Computed advantages (n_steps,)
        returns: Computed returns (n_steps,)
    """
    states, actions, logprobs, values, rewards = [], [], [], [], []
    
    # Reset environment
    state, info = env.reset()
    done = False
    step = 0
    
    agent.train()  # Set to training mode
    
    while not done and step < max_steps:
        # Convert state to tensor
        state_t = torch.FloatTensor(state).unsqueeze(0)
        
        # Get action from policy
        action_logits, value = agent(state_t)
        action_probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        action = dist.sample()
        logp = dist.log_prob(action)
        
        # Take action in environment
        next_state, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated
        
        # Store trajectory data
        states.append(state_t)
        actions.append(action)
        logprobs.append(logp)
        values.append(value)
        rewards.append(reward)
        
        state = next_state
        step += 1
    
    # Convert to tensors
    if len(states) == 0:
        # Empty episode
        dummy_state = torch.zeros((1, env.observation_space.shape[0]))
        return 0.0, dummy_state, torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])
    
    states = torch.cat(states)
    actions = torch.cat(actions)
    logprobs = torch.cat(logprobs)
    values = torch.cat(values).squeeze(-1)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    
    # Compute returns (discounted cumulative rewards)
    returns = compute_returns(rewards, gamma)
    
    # Normalize returns for stability (only if std > 0)
    if returns.std() > 1e-8:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    # Compute advantages using GAE
    advantages = compute_gae_advantages(rewards, values, gamma, lambda_gae=0.95)
    
    # Normalize advantages (only if std > 0)
    if advantages.std() > 1e-8:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    total_reward = rewards.sum().item()
    
    return total_reward, states, actions, logprobs, advantages, returns


def compute_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """Compute discounted returns
    
    Formula: G_t = r_t + γ * G_{t+1}
    """
    returns = []
    G = 0
    
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    
    return torch.tensor(returns, dtype=torch.float32)


def compute_gae_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lambda_gae: float = 0.95
) -> torch.Tensor:
    """Compute advantages using Generalized Advantage Estimation (GAE)
    
    Formula:
    δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
    Â_t = δ_t + (γ * λ) * Â_{t+1}
    
    This reduces variance compared to simple advantage estimation.
    """
    if len(rewards) == 0:
        return torch.tensor([], dtype=torch.float32)
    
    advantages = []
    gae = 0
    
    # Append bootstrap value for last step
    next_value = 0  # Assuming episode ended
    values_plus_next = torch.cat([values, torch.tensor([next_value])])
    
    for t in reversed(range(len(rewards))):
        # TD error
        delta = rewards[t] + gamma * values_plus_next[t + 1] - values_plus_next[t]
        
        # GAE
        gae = delta + gamma * lambda_gae * gae
        advantages.insert(0, gae)
    
    return torch.tensor(advantages, dtype=torch.float32)

