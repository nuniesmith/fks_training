"""
Evaluate trained PPO model

Usage:
    python -m training.src.ppo.evaluate_model \
        --model-path ./models/ppo/ppo_meta_learning.pt \
        --ticker AAPL \
        --start-date 2024-01-01 \
        --end-date 2025-01-01 \
        --n-episodes 10
"""
import argparse
import torch
from pathlib import Path
from loguru import logger

from .trading_env import TradingEnv
from .policy_network import DualHeadPPOPolicy
from .evaluation import PPOEvaluator


def load_model(model_path: str, feature_dim: int = 22, num_actions: int = 3) -> DualHeadPPOPolicy:
    """Load trained PPO model"""
    logger.info(f"Loading model from {model_path}")
    
    policy = DualHeadPPOPolicy(
        feature_dim=feature_dim,
        hidden_dim=128,
        num_actions=num_actions,
        dropout=0.2
    )
    
    try:
        state_dict = torch.load(model_path, map_location="cpu")
        policy.load_state_dict(state_dict)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
    
    return policy


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained PPO model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol")
    parser.add_argument("--start-date", type=str, default="2024-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="2025-01-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--data-source", type=str, default="yfinance", choices=["yfinance", "fks_data"], help="Data source")
    parser.add_argument("--n-episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--initial-balance", type=float, default=10000.0, help="Initial cash balance")
    parser.add_argument("--transaction-cost", type=float, default=0.001, help="Transaction cost")
    parser.add_argument("--output-report", type=str, help="Path to save evaluation report")
    parser.add_argument("--compare-baseline", action="store_true", help="Compare with baseline strategies")
    
    args = parser.parse_args()
    
    # Load model
    try:
        model = load_model(args.model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Create environment
    logger.info(f"Creating trading environment for {args.ticker}...")
    try:
        env = TradingEnv(
            symbol=args.ticker,
            start_date=args.start_date,
            end_date=args.end_date,
            initial_balance=args.initial_balance,
            transaction_cost=args.transaction_cost,
            data_source=args.data_source,
            normalize_states=True
        )
        logger.info("Environment created successfully")
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        return
    
    # Create evaluator
    evaluator = PPOEvaluator(model, env)
    
    # Evaluate
    logger.info(f"Evaluating model on {args.n_episodes} episodes...")
    metrics = evaluator.evaluate_performance(n_episodes=args.n_episodes)
    
    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Episodes: {metrics.get('n_episodes', 0)}")
    print(f"Average Return: {metrics.get('avg_return', 0):.2%}")
    print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
    print(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
    print(f"Directional Accuracy: {metrics.get('directional_accuracy', 0):.2%}")
    print("=" * 60)
    
    # Compare with baseline if requested
    if args.compare_baseline:
        logger.info("Comparing with baseline strategies...")
        comparison = evaluator.compare_with_baseline(n_episodes=args.n_episodes)
        
        print("\n" + "=" * 60)
        print("Baseline Comparison")
        print("=" * 60)
        improvement = comparison.get("improvement", {})
        print(f"Return Improvement: {improvement.get('return', 0):.2%}")
        print(f"Sharpe Improvement: {improvement.get('sharpe', 0):.2f}")
        print(f"Accuracy Improvement: {improvement.get('accuracy', 0):.2%}")
        print("=" * 60)
    
    # Generate report if requested
    if args.output_report:
        logger.info(f"Generating report to {args.output_report}...")
        report = evaluator.generate_report(
            output_path=args.output_report,
            n_episodes=args.n_episodes
        )
        print(f"\nReport saved to {args.output_report}")
    
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()

