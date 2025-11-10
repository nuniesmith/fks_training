"""
PPO Model Evaluation Framework

Comprehensive evaluation framework for PPO trading models including:
- Performance metrics (Sharpe ratio, drawdown, returns)
- Directional accuracy
- Action distribution analysis
- Baseline comparison
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger
from pathlib import Path

from .policy_network import DualHeadPPOPolicy
from .trading_env import TradingEnv


class PPOEvaluator:
    """Comprehensive evaluation framework for PPO trading models"""
    
    def __init__(self, model: DualHeadPPOPolicy, env: TradingEnv):
        """
        Initialize PPO evaluator.
        
        Args:
            model: Trained PPO policy network
            env: Trading environment for evaluation
        """
        self.model = model
        self.env = env
        self.model.eval()  # Set to evaluation mode
    
    def evaluate_performance(
        self,
        n_episodes: int = 10,
        deterministic: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate model performance on trading environment.
        
        Args:
            n_episodes: Number of episodes to evaluate
            deterministic: Use deterministic policy
        
        Returns:
            Dictionary with performance metrics
        """
        logger.info(f"Evaluating PPO model on {n_episodes} episodes...")
        
        episode_results = []
        all_actions = []
        all_rewards = []
        all_returns = []
        all_positions = []
        
        for episode in range(n_episodes):
            try:
                state, info = self.env.reset()
                done = False
                episode_rewards = []
                episode_actions = []
                episode_returns = []
                episode_positions = []
                step = 0
                
                initial_balance = info.get("balance", 10000.0)
                
                while not done and step < 1000:
                    # Get action from policy
                    import torch
                    state_t = torch.FloatTensor(state).unsqueeze(0)
                    action, value, _, _ = self.model.get_action(state_t, deterministic=deterministic)
                    
                    # Step environment
                    next_state, reward, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated
                    
                    episode_rewards.append(reward)
                    episode_actions.append(action)
                    episode_returns.append(info.get("returns", 0.0))
                    episode_positions.append(info.get("position", 0))
                    
                    state = next_state
                    step += 1
                
                final_balance = info.get("balance", initial_balance)
                total_return = (final_balance - initial_balance) / initial_balance
                
                episode_results.append({
                    "episode": episode,
                    "total_reward": sum(episode_rewards),
                    "total_return": total_return,
                    "final_balance": final_balance,
                    "initial_balance": initial_balance,
                    "steps": step,
                    "actions": episode_actions,
                    "rewards": episode_rewards,
                    "returns": episode_returns,
                    "positions": episode_positions
                })
                
                all_actions.extend(episode_actions)
                all_rewards.extend(episode_rewards)
                all_returns.extend(episode_returns)
                all_positions.extend(episode_positions)
                
            except Exception as e:
                logger.warning(f"Error in evaluation episode {episode}: {e}")
                continue
        
        if not episode_results:
            logger.error("No successful evaluation episodes")
            return {
                "error": "No successful episodes",
                "n_episodes": 0
            }
        
        # Calculate metrics
        metrics = self._calculate_metrics(episode_results, all_actions, all_rewards, all_returns)
        
        logger.info(f"Evaluation complete: {metrics.get('avg_return', 0):.2%} avg return, "
                   f"{metrics.get('sharpe_ratio', 0):.2f} Sharpe ratio")
        
        return metrics
    
    def _calculate_metrics(
        self,
        episode_results: List[Dict[str, Any]],
        all_actions: List[int],
        all_rewards: List[float],
        all_returns: List[float]
    ) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        
        # Basic statistics
        total_returns = [r["total_return"] for r in episode_results]
        total_rewards = [r["total_reward"] for r in episode_results]
        
        avg_return = np.mean(total_returns)
        std_return = np.std(total_returns)
        avg_reward = np.mean(total_rewards)
        
        # Sharpe ratio (annualized, assuming daily returns)
        if std_return > 0:
            sharpe_ratio = (avg_return / std_return) * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0.0
        
        # Maximum drawdown
        max_drawdown = self._calculate_max_drawdown(total_returns)
        
        # Win rate
        winning_episodes = sum(1 for r in total_returns if r > 0)
        win_rate = winning_episodes / len(total_returns) if total_returns else 0.0
        
        # Directional accuracy (if we can infer from actions/returns)
        directional_accuracy = self._calculate_directional_accuracy(episode_results)
        
        # Action distribution
        action_distribution = self._calculate_action_distribution(all_actions)
        
        # Returns distribution
        returns_stats = {
            "mean": np.mean(all_returns) if all_returns else 0.0,
            "std": np.std(all_returns) if all_returns else 0.0,
            "min": np.min(all_returns) if all_returns else 0.0,
            "max": np.max(all_returns) if all_returns else 0.0,
            "median": np.median(all_returns) if all_returns else 0.0
        }
        
        return {
            "n_episodes": len(episode_results),
            "avg_return": avg_return,
            "std_return": std_return,
            "avg_reward": avg_reward,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "directional_accuracy": directional_accuracy,
            "action_distribution": action_distribution,
            "returns_stats": returns_stats,
            "episode_results": episode_results
        }
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown from returns"""
        if not returns:
            return 0.0
        
        cumulative = np.cumprod([1 + r for r in returns])
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return abs(np.min(drawdown))
    
    def _calculate_directional_accuracy(
        self,
        episode_results: List[Dict[str, Any]]
    ) -> float:
        """Calculate directional accuracy from episode results"""
        if not episode_results:
            return 0.0
        
        correct_predictions = 0
        total_predictions = 0
        
        for result in episode_results:
            returns = result.get("returns", [])
            actions = result.get("actions", [])
            
            # Simple heuristic: if action is buy (1) and return is positive, correct
            # If action is sell (2) and return is negative, correct
            for i in range(min(len(returns), len(actions))):
                if i == 0:
                    continue
                
                action = actions[i-1]
                return_val = returns[i] if i < len(returns) else 0.0
                
                # Buy action (1) should predict positive return
                # Sell action (2) should predict negative return
                # Hold action (0) doesn't count
                if action == 1 and return_val > 0:
                    correct_predictions += 1
                elif action == 2 and return_val < 0:
                    correct_predictions += 1
                
                if action != 0:  # Only count non-hold actions
                    total_predictions += 1
        
        if total_predictions == 0:
            return 0.0
        
        return correct_predictions / total_predictions
    
    def _calculate_action_distribution(self, actions: List[int]) -> Dict[str, float]:
        """Calculate action distribution"""
        if not actions:
            return {"hold": 0.0, "buy": 0.0, "sell": 0.0}
        
        action_counts = {}
        for action in actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        total = len(actions)
        return {
            "hold": action_counts.get(0, 0) / total,
            "buy": action_counts.get(1, 0) / total,
            "sell": action_counts.get(2, 0) / total
        }
    
    def compare_with_baseline(
        self,
        baseline_strategy: str = "buy_and_hold",
        n_episodes: int = 10
    ) -> Dict[str, Any]:
        """
        Compare PPO model performance with baseline strategies.
        
        Args:
            baseline_strategy: Baseline strategy ("buy_and_hold", "random", "momentum")
            n_episodes: Number of episodes for comparison
        
        Returns:
            Dictionary with comparison results
        """
        logger.info(f"Comparing PPO model with {baseline_strategy} baseline...")
        
        # Evaluate PPO model
        ppo_metrics = self.evaluate_performance(n_episodes=n_episodes)
        
        # Evaluate baseline
        baseline_metrics = self._evaluate_baseline(baseline_strategy, n_episodes)
        
        # Calculate improvement
        improvement = {}
        if "avg_return" in ppo_metrics and "avg_return" in baseline_metrics:
            improvement["return"] = ppo_metrics["avg_return"] - baseline_metrics["avg_return"]
            improvement["return_pct"] = (
                (ppo_metrics["avg_return"] - baseline_metrics["avg_return"]) /
                abs(baseline_metrics["avg_return"]) * 100
                if baseline_metrics["avg_return"] != 0 else 0.0
            )
        
        if "sharpe_ratio" in ppo_metrics and "sharpe_ratio" in baseline_metrics:
            improvement["sharpe"] = ppo_metrics["sharpe_ratio"] - baseline_metrics["sharpe_ratio"]
        
        if "directional_accuracy" in ppo_metrics and "directional_accuracy" in baseline_metrics:
            improvement["accuracy"] = (
                ppo_metrics["directional_accuracy"] - baseline_metrics["directional_accuracy"]
            )
        
        return {
            "ppo_metrics": ppo_metrics,
            "baseline_metrics": baseline_metrics,
            "improvement": improvement,
            "baseline_strategy": baseline_strategy
        }
    
    def _evaluate_baseline(
        self,
        strategy: str,
        n_episodes: int
    ) -> Dict[str, Any]:
        """Evaluate baseline strategy"""
        episode_results = []
        
        for episode in range(n_episodes):
            try:
                state, info = self.env.reset()
                done = False
                step = 0
                initial_balance = info.get("balance", 10000.0)
                
                while not done and step < 1000:
                    # Baseline strategy actions
                    if strategy == "buy_and_hold":
                        action = 1  # Always buy
                    elif strategy == "random":
                        action = np.random.randint(0, 3)  # Random action
                    elif strategy == "momentum":
                        # Simple momentum: buy if price going up, sell if going down
                        if len(state) > 0:
                            # Use price trend from state
                            action = 1 if state[0] > 0 else 2
                        else:
                            action = 0
                    else:
                        action = 0  # Hold
                    
                    next_state, reward, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated
                    state = next_state
                    step += 1
                
                final_balance = info.get("balance", initial_balance)
                total_return = (final_balance - initial_balance) / initial_balance
                
                episode_results.append({
                    "total_return": total_return,
                    "total_reward": sum([reward])  # Simplified
                })
                
            except Exception as e:
                logger.warning(f"Error in baseline evaluation episode {episode}: {e}")
                continue
        
        if not episode_results:
            return {"avg_return": 0.0, "sharpe_ratio": 0.0, "directional_accuracy": 0.0}
        
        total_returns = [r["total_return"] for r in episode_results]
        avg_return = np.mean(total_returns)
        std_return = np.std(total_returns)
        
        sharpe_ratio = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0.0
        
        return {
            "avg_return": avg_return,
            "std_return": std_return,
            "sharpe_ratio": sharpe_ratio,
            "directional_accuracy": 0.5  # Baseline assumption
        }
    
    def generate_report(
        self,
        output_path: Optional[str] = None,
        n_episodes: int = 10
    ) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            output_path: Path to save report (optional)
            n_episodes: Number of episodes for evaluation
        
        Returns:
            Report text
        """
        logger.info("Generating evaluation report...")
        
        # Evaluate performance
        metrics = self.evaluate_performance(n_episodes=n_episodes)
        
        # Compare with baseline
        comparison = self.compare_with_baseline(n_episodes=n_episodes)
        
        # Generate report
        report_lines = [
            "=" * 60,
            "PPO Trading Model Evaluation Report",
            "=" * 60,
            "",
            f"Evaluation Episodes: {metrics.get('n_episodes', 0)}",
            "",
            "Performance Metrics:",
            f"  Average Return: {metrics.get('avg_return', 0):.2%}",
            f"  Standard Deviation: {metrics.get('std_return', 0):.2%}",
            f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}",
            f"  Maximum Drawdown: {metrics.get('max_drawdown', 0):.2%}",
            f"  Win Rate: {metrics.get('win_rate', 0):.2%}",
            f"  Directional Accuracy: {metrics.get('directional_accuracy', 0):.2%}",
            "",
            "Action Distribution:",
        ]
        
        action_dist = metrics.get("action_distribution", {})
        for action, pct in action_dist.items():
            report_lines.append(f"  {action.capitalize()}: {pct:.2%}")
        
        report_lines.extend([
            "",
            "Baseline Comparison:",
            f"  Baseline Strategy: {comparison.get('baseline_strategy', 'N/A')}",
        ])
        
        improvement = comparison.get("improvement", {})
        if improvement:
            report_lines.append(f"  Return Improvement: {improvement.get('return', 0):.2%}")
            report_lines.append(f"  Sharpe Improvement: {improvement.get('sharpe', 0):.2f}")
            report_lines.append(f"  Accuracy Improvement: {improvement.get('accuracy', 0):.2%}")
        
        report_lines.extend([
            "",
            "=" * 60
        ])
        
        report_text = "\n".join(report_lines)
        
        # Save report if path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(report_text)
            logger.info(f"Report saved to {output_path}")
        
        return report_text

