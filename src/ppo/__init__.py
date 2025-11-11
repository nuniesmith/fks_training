"""PPO Meta-Learning Module for FKS Trading"""

from .networks import BackboneNetwork
from .policy_network import DualHeadPPOPolicy
from .feature_extractor import FKSFeatureExtractor
from .trading_env import TradingEnv
from .trainer import PPOTrainer
from .data_collection import forward_pass, compute_returns, compute_gae_advantages
from .training_loop import evaluate, run_ppo_training

# Optional evaluation import
try:
    from .evaluation import PPOEvaluator
    __all__ = [
        "BackboneNetwork",
        "DualHeadPPOPolicy",
        "FKSFeatureExtractor",
        "TradingEnv",
        "PPOTrainer",
        "forward_pass",
        "compute_returns",
        "compute_gae_advantages",
        "evaluate",
        "run_ppo_training",
        "PPOEvaluator"
    ]
except ImportError:
    __all__ = [
        "BackboneNetwork",
        "DualHeadPPOPolicy",
        "FKSFeatureExtractor",
        "TradingEnv",
        "PPOTrainer",
        "forward_pass",
        "compute_returns",
        "compute_gae_advantages",
        "evaluate",
        "run_ppo_training"
    ]
