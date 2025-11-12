"""
Train Reinforcement Learning Agent for Portfolio Management

Uses Stable Baselines3 to train PPO agent on PortfolioEnv.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from domain.ml.rl.environment import PortfolioEnv, PortfolioEnvConfig


def load_portfolio_data(data_path: str) -> pd.DataFrame:
    """
    Load price data for portfolio environment.

    Args:
        data_path: Path to CSV file with price data

    Returns:
        DataFrame with asset prices
    """
    df = pd.read_csv(data_path)
    
    # Ensure timestamp column if present
    if "timestamp" not in df.columns and "date" in df.columns:
        df.rename(columns={"date": "timestamp"}, inplace=True)
    
    return df


def main():
    """Main RL training function."""
    parser = argparse.ArgumentParser(description="Train RL agent for portfolio management")
    parser.add_argument("--data-path", type=str, required=True, help="Path to price data CSV")
    parser.add_argument("--model-name", type=str, default="ppo_portfolio")
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial-balance", type=float, default=10000.0)
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--reward-type", type=str, default="sharpe", choices=["sharpe", "returns", "risk_adjusted"])
    parser.add_argument("--tracking-uri", type=str, help="MLflow tracking URI")

    args = parser.parse_args()

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)

    # Load data
    print(f"Loading data from {args.data_path}...")
    price_data = load_portfolio_data(args.data_path)

    # Create environment config
    env_config = PortfolioEnvConfig(
        initial_balance=args.initial_balance,
        transaction_cost=args.transaction_cost,
        reward_type=args.reward_type,
    )

    # Create environments
    train_env = PortfolioEnv(price_data, config=env_config, action_type="continuous")
    train_env = Monitor(train_env)  # Wrap for logging

    eval_env = PortfolioEnv(price_data, config=env_config, action_type="continuous")
    eval_env = Monitor(eval_env)

    # Create PPO agent
    print("Creating PPO agent...")
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        verbose=1,
        tensorboard_log="./logs/tensorboard/",
    )

    # Setup MLflow tracking
    with mlflow.start_run(run_name=f"{args.model_name}_rl"):
        # Log hyperparameters
        mlflow.log_params(
            {
                "algorithm": "PPO",
                "total_timesteps": args.total_timesteps,
                "learning_rate": args.learning_rate,
                "n_steps": args.n_steps,
                "batch_size": args.batch_size,
                "n_epochs": args.n_epochs,
                "gamma": args.gamma,
                "initial_balance": args.initial_balance,
                "transaction_cost": args.transaction_cost,
                "reward_type": args.reward_type,
                "n_assets": train_env.n_assets,
            }
        )

        # Setup evaluation callback
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path="./models/rl/",
            log_path="./logs/eval/",
            eval_freq=5000,
            deterministic=True,
            render=False,
        )

        # Train agent
        print(f"Training PPO agent for {args.total_timesteps} timesteps...")
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=eval_callback,
            progress_bar=True,
        )

        # Evaluate final performance
        print("Evaluating final performance...")
        obs, _ = eval_env.reset()
        episode_rewards = []
        episode_lengths = []

        for _ in range(10):  # Run 10 episodes
            episode_reward = 0
            episode_length = 0
            done = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated

                if done:
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(episode_length)
                    obs, _ = eval_env.reset()

        # Log metrics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)

        mlflow.log_metric("mean_episode_reward", mean_reward)
        mlflow.log_metric("std_episode_reward", std_reward)
        mlflow.log_metric("mean_episode_length", mean_length)

        # Log final portfolio performance
        final_portfolio_value = eval_env.portfolio_value
        total_return = (final_portfolio_value - args.initial_balance) / args.initial_balance

        mlflow.log_metric("final_portfolio_value", final_portfolio_value)
        mlflow.log_metric("total_return", total_return)

        # Save model
        model_path = f"./models/rl/{args.model_name}_final"
        model.save(model_path)
        mlflow.log_artifact(model_path + ".zip", "model")

        print(f"\nTraining completed!")
        print(f"Mean Episode Reward: {mean_reward:.4f} ± {std_reward:.4f}")
        print(f"Total Return: {total_return * 100:.2f}%")
        print(f"Final Portfolio Value: ${final_portfolio_value:.2f}")


if __name__ == "__main__":
    main()

