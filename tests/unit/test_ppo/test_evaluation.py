"""
Tests for PPO Evaluation Framework
"""
import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.ppo.policy_network import DualHeadPPOPolicy
from src.ppo.trading_env import TradingEnv
from src.ppo.evaluation import PPOEvaluator


class TestPPOEvaluator:
    """Test PPO evaluator"""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock PPO model"""
        model = Mock(spec=DualHeadPPOPolicy)
        model.eval = MagicMock()
        model.train = MagicMock()
        model.get_action = MagicMock(return_value=(1, 0.5, torch.tensor(0.0), None))
        return model
    
    @pytest.fixture
    def mock_env(self):
        """Create mock trading environment"""
        env = Mock(spec=TradingEnv)
        env.reset = MagicMock(return_value=(np.zeros(22), {"balance": 10000.0}))
        env.step = MagicMock(return_value=(np.zeros(22), 1.0, False, False, {"balance": 10100.0, "returns": 0.01}))
        return env
    
    @pytest.fixture
    def evaluator(self, mock_model, mock_env):
        """Create PPO evaluator"""
        return PPOEvaluator(mock_model, mock_env)
    
    def test_evaluator_initialization(self, mock_model, mock_env):
        """Test evaluator initialization"""
        evaluator = PPOEvaluator(mock_model, mock_env)
        assert evaluator.model == mock_model
        assert evaluator.env == mock_env
        mock_model.eval.assert_called_once()
    
    def test_evaluate_performance(self, evaluator, mock_model, mock_env):
        """Test performance evaluation"""
        # Mock environment to return done after a few steps
        step_count = [0]
        def mock_step(action):
            step_count[0] += 1
            done = step_count[0] >= 5
            return (np.zeros(22), 1.0, done, False, {"balance": 10100.0, "returns": 0.01, "position": 1})
        
        mock_env.step = MagicMock(side_effect=mock_step)
        
        metrics = evaluator.evaluate_performance(n_episodes=2, deterministic=True)
        
        assert "n_episodes" in metrics
        assert "avg_return" in metrics
        assert "sharpe_ratio" in metrics
        assert metrics["n_episodes"] > 0
    
    def test_calculate_max_drawdown(self, evaluator):
        """Test maximum drawdown calculation"""
        returns = [0.01, -0.02, 0.01, -0.03, 0.02]
        drawdown = evaluator._calculate_max_drawdown(returns)
        
        assert drawdown >= 0.0
        assert isinstance(drawdown, float)
    
    def test_calculate_directional_accuracy(self, evaluator):
        """Test directional accuracy calculation"""
        episode_results = [
            {
                "returns": [0.01, 0.02, -0.01],
                "actions": [1, 1, 2]  # Buy, Buy, Sell
            }
        ]
        
        accuracy = evaluator._calculate_directional_accuracy(episode_results)
        
        assert 0.0 <= accuracy <= 1.0
        assert isinstance(accuracy, float)
    
    def test_calculate_action_distribution(self, evaluator):
        """Test action distribution calculation"""
        actions = [0, 0, 1, 1, 2]  # Hold, Hold, Buy, Buy, Sell
        
        distribution = evaluator._calculate_action_distribution(actions)
        
        assert "hold" in distribution
        assert "buy" in distribution
        assert "sell" in distribution
        assert sum(distribution.values()) == pytest.approx(1.0)
    
    def test_compare_with_baseline(self, evaluator, mock_model, mock_env):
        """Test baseline comparison"""
        # Mock evaluation methods
        evaluator.evaluate_performance = MagicMock(return_value={
            "n_episodes": 2,
            "avg_return": 0.05,
            "sharpe_ratio": 1.0,
            "directional_accuracy": 0.65
        })
        
        evaluator._evaluate_baseline = MagicMock(return_value={
            "avg_return": 0.03,
            "sharpe_ratio": 0.5,
            "directional_accuracy": 0.50
        })
        
        comparison = evaluator.compare_with_baseline(baseline_strategy="buy_and_hold", n_episodes=2)
        
        assert "ppo_metrics" in comparison
        assert "baseline_metrics" in comparison
        assert "improvement" in comparison
        assert comparison["baseline_strategy"] == "buy_and_hold"
    
    def test_generate_report(self, evaluator):
        """Test report generation"""
        # Mock evaluation methods
        evaluator.evaluate_performance = MagicMock(return_value={
            "n_episodes": 2,
            "avg_return": 0.05,
            "std_return": 0.02,
            "sharpe_ratio": 1.0,
            "max_drawdown": 0.03,
            "win_rate": 0.6,
            "directional_accuracy": 0.65,
            "action_distribution": {"hold": 0.3, "buy": 0.5, "sell": 0.2}
        })
        
        evaluator.compare_with_baseline = MagicMock(return_value={
            "baseline_strategy": "buy_and_hold",
            "improvement": {"return": 0.02, "sharpe": 0.5, "accuracy": 0.15}
        })
        
        report = evaluator.generate_report(n_episodes=2)
        
        assert isinstance(report, str)
        assert "PPO Trading Model Evaluation Report" in report
        assert "Average Return" in report
        assert "Sharpe Ratio" in report

