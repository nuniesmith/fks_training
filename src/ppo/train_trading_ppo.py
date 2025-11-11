"""
Complete PPO training script for stock trading

Usage:
    python -m training.src.ppo.train_trading_ppo \
        --ticker AAPL \
        --start-date 2020-01-01 \
        --end-date 2025-11-01 \
        --max-episodes 1000 \
        --data-source yfinance
"""
import argparse
import torch
import numpy as np
from loguru import logger
from pathlib import Path

from .trading_env import TradingEnv
from .training_loop import evaluate, run_ppo_training


def split_data_dates(start_date: str, end_date: str, train_ratio: float = 0.8):
    """Split date range into train and test periods"""
    from datetime import datetime, timedelta
    
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end - start).days
    train_days = int(total_days * train_ratio)
    
    train_end = start + timedelta(days=train_days)
    test_start = train_end + timedelta(days=1)
    
    return (
        start_date,
        train_end.strftime("%Y-%m-%d"),
        test_start.strftime("%Y-%m-%d"),
        end_date
    )


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent for stock trading")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol")
    parser.add_argument("--start-date", type=str, default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="2025-11-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--data-source", type=str, default="yfinance", choices=["yfinance", "fks_data"], help="Data source")
    parser.add_argument("--max-episodes", type=int, default=1000, help="Maximum training episodes")
    parser.add_argument("--initial-balance", type=float, default=10000.0, help="Initial cash balance")
    parser.add_argument("--transaction-cost", type=float, default=0.001, help="Transaction cost (0.001 = 0.1%%)")
    parser.add_argument("--slippage", type=float, default=0.0005, help="Slippage (0.0005 = 0.05%%)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clipping parameter")
    parser.add_argument("--ppo-epochs", type=int, default=10, help="PPO update epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--threshold", type=float, default=0.5, help="Early stopping threshold (reward)")
    parser.add_argument("--model-save-path", type=str, default="./models/ppo/trading_ppo.pt", help="Model save path")
    parser.add_argument("--use-mlflow", action="store_true", help="Use MLflow for tracking")
    parser.add_argument("--mlflow-uri", type=str, help="MLflow tracking URI")
    
    args = parser.parse_args()
    
    # Setup MLflow
    if args.use_mlflow:
        try:
            import mlflow
            if args.mlflow_uri:
                mlflow.set_tracking_uri(args.mlflow_uri)
            mlflow.set_experiment("ppo_trading")
        except ImportError:
            logger.warning("MLflow not available, skipping tracking")
            args.use_mlflow = False
    
    # Split data into train and test
    train_start, train_end, test_start, test_end = split_data_dates(
        args.start_date,
        args.end_date,
        train_ratio=0.8
    )
    
    logger.info(f"Training period: {train_start} to {train_end}")
    logger.info(f"Test period: {test_start} to {test_end}")
    
    # Create training environment
    logger.info(f"Creating training environment for {args.ticker}...")
    try:
        train_env = TradingEnv(
            ticker=args.ticker,
            start_date=train_start,
            end_date=train_end,
            initial_balance=args.initial_balance,
            transaction_cost=args.transaction_cost,
            slippage=args.slippage,
            data_source=args.data_source,
            normalize_states=True
        )
    except Exception as e:
        logger.error(f"Failed to create training environment: {e}")
        return
    
    # Create test environment
    logger.info(f"Creating test environment for {args.ticker}...")
    try:
        test_env = TradingEnv(
            ticker=args.ticker,
            start_date=test_start,
            end_date=test_end,
            initial_balance=args.initial_balance,
            transaction_cost=args.transaction_cost,
            slippage=args.slippage,
            data_source=args.data_source,
            normalize_states=True
        )
    except Exception as e:
        logger.error(f"Failed to create test environment: {e}")
        return
    
    # Get feature dimension from environment
    obs_shape = train_env.observation_space.shape
    feature_dim = obs_shape[0] if len(obs_shape) == 1 else obs_shape[0]
    num_actions = 3  # hold, buy, sell
    
    logger.info(f"Feature dimension: {feature_dim}")
    logger.info(f"Action space: {num_actions} (hold, buy, sell)")
    
    # Train PPO agent
    logger.info("Starting PPO training...")
    results = run_ppo_training(
        env_train=train_env,
        env_test=test_env,
        feature_dim=feature_dim,
        num_actions=num_actions,
        max_episodes=args.max_episodes,
        threshold=args.threshold,
        lr=args.lr,
        gamma=args.gamma,
        epsilon=args.epsilon,
        ppo_epochs=args.ppo_epochs,
        batch_size=args.batch_size,
        model_save_path=args.model_save_path,
        use_mlflow=args.use_mlflow
    )
    
    # Final evaluation on test set
    logger.info("Running final evaluation on test set...")
    policy = results["policy"]
    final_reward = evaluate(test_env, policy, n_episodes=10, deterministic=True)
    
    logger.info(f"Final test reward: {final_reward:.4f}")
    logger.info(f"Best test reward: {results['training_stats']['best_test_reward']:.4f}")
    logger.info(f"Model saved to: {results['model_path']}")
    
    # Calculate performance metrics
    episode_rewards = []
    episode_returns = []
    
    for _ in range(10):
        try:
            obs, info = test_env.reset()
            done = False
            total_reward = 0.0
            
            while not done:
                obs_t = torch.FloatTensor(obs).unsqueeze(0)
                action, _, _, _ = policy.get_action(obs_t, deterministic=True)
                obs, reward, terminated, truncated, info = test_env.step(action)
                done = terminated or truncated
                total_reward += reward
            
            episode_rewards.append(total_reward)
            final_info = test_env._get_info()
            episode_returns.append(final_info["profit_pct"])
        except Exception as e:
            logger.warning(f"Error in evaluation episode: {e}")
            continue
    
    if episode_returns:
        avg_return = np.mean(episode_returns)
        sharpe_ratio = np.mean(episode_returns) / (np.std(episode_returns) + 1e-8) * np.sqrt(252)  # Annualized
        
        logger.info(f"Average return: {avg_return:.2f}%")
        logger.info(f"Sharpe ratio: {sharpe_ratio:.2f}")
        
        if args.use_mlflow:
            try:
                import mlflow
                mlflow.log_metrics({
                    "final_test_reward": final_reward,
                    "avg_return": avg_return,
                    "sharpe_ratio": sharpe_ratio
                })
            except Exception as e:
                logger.warning(f"Failed to log metrics to MLflow: {e}")
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()

