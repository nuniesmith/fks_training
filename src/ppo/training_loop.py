"""
Complete PPO Training Loop

Training loop with evaluation, MLflow integration, and early stopping.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional
from loguru import logger
from pathlib import Path

from .policy_network import DualHeadPPOPolicy
from .trainer import PPOTrainer
from .data_collection import forward_pass


def evaluate(
    env,
    agent: DualHeadPPOPolicy,
    n_episodes: int = 10,
    deterministic: bool = True
) -> float:
    """Evaluate agent performance
    
    Args:
        env: Trading environment
        agent: PPO policy network
        n_episodes: Number of evaluation episodes
        deterministic: If True, use deterministic policy (no exploration)
    
    Returns:
        Average episode reward
    """
    agent.eval()  # Set to evaluation mode
    episode_rewards = []
    
    with torch.no_grad():
        for _ in range(n_episodes):
            try:
                state, info = env.reset()
                done = False
                total_reward = 0.0
                step = 0
                
                while not done and step < 1000:
                    state_t = torch.FloatTensor(state).unsqueeze(0)
                    action, value, _, _ = agent.get_action(state_t, deterministic=deterministic)
                    
                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    total_reward += reward
                    state = next_state
                    step += 1
                
                episode_rewards.append(total_reward)
            except Exception as e:
                logger.warning(f"Error in evaluation episode: {e}")
                continue
    
    agent.train()  # Reset to training mode
    return np.mean(episode_rewards) if episode_rewards else 0.0


def run_ppo_training(
    env_train,
    env_test,
    feature_dim: int = 22,
    num_actions: int = 10,
    max_episodes: int = 1000,
    threshold: float = 475.0,  # Reward threshold for early stopping
    lr: float = 0.001,
    gamma: float = 0.99,
    epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    ppo_epochs: int = 10,
    batch_size: int = 128,
    max_grad_norm: float = 0.5,
    log_interval: int = 10,
    save_interval: int = 100,
    model_save_path: Optional[str] = None,
    use_mlflow: bool = True
) -> Dict[str, Any]:
    """Complete PPO training loop
    
    Training Process:
    1. Initialize policy and trainer
    2. For each episode:
       a. Collect trajectory using current policy
       b. Compute returns and advantages (GAE)
       c. Update policy using PPO (multiple epochs on batches)
       d. Evaluate on test environment
       e. Log metrics
       f. Save model periodically
    3. Early stop if threshold reached
    
    Args:
        env_train: Training environment
        env_test: Test environment for evaluation
        feature_dim: Dimension of feature vector (22D)
        num_actions: Number of actions (strategies/models)
        max_episodes: Maximum training episodes
        threshold: Reward threshold for early stopping
        lr: Learning rate
        gamma: Discount factor
        epsilon: PPO clipping parameter
        value_coef: Value loss coefficient
        entropy_coef: Entropy coefficient
        ppo_epochs: Number of PPO update epochs
        batch_size: Batch size for updates
        max_grad_norm: Maximum gradient norm for clipping
        log_interval: Log metrics every N episodes
        save_interval: Save model every N episodes
        model_save_path: Path to save model
        use_mlflow: Whether to use MLflow for tracking
    
    Returns:
        Training statistics and final model
    """
    # Initialize policy and trainer
    logger.info("Initializing PPO policy and trainer...")
    policy = DualHeadPPOPolicy(
        feature_dim=feature_dim,
        hidden_dim=128,
        num_actions=num_actions,
        dropout=0.2
    )
    
    trainer = PPOTrainer(
        policy=policy,
        lr=lr,
        gamma=gamma,
        epsilon=epsilon,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        max_grad_norm=max_grad_norm,
        ppo_epochs=ppo_epochs,
        batch_size=batch_size
    )
    
    # MLflow tracking (optional)
    mlflow_run = None
    if use_mlflow:
        try:
            import mlflow
            mlflow.set_experiment("ppo_meta_learning")
            mlflow_run = mlflow.start_run()
            mlflow.log_params({
                "feature_dim": feature_dim,
                "num_actions": num_actions,
                "lr": lr,
                "gamma": gamma,
                "epsilon": epsilon,
                "value_coef": value_coef,
                "entropy_coef": entropy_coef,
                "ppo_epochs": ppo_epochs,
                "batch_size": batch_size,
                "max_grad_norm": max_grad_norm
            })
        except ImportError:
            logger.warning("MLflow not available, skipping tracking")
            use_mlflow = False
    
    # Training statistics
    training_stats = {
        "episode_rewards": [],
        "test_rewards": [],
        "policy_losses": [],
        "value_losses": [],
        "entropies": [],
        "best_test_reward": -np.inf,
        "episodes_trained": 0
    }
    
    # Model save path
    if model_save_path is None:
        model_save_path = "./models/ppo/ppo_meta_learning.pt"
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting PPO training for {max_episodes} episodes...")
    logger.info(f"Early stop threshold: {threshold}")
    
    # Training loop
    for episode in range(1, max_episodes + 1):
        try:
            # Collect trajectory
            train_reward, states, actions, old_log_probs, advantages, returns = forward_pass(
                env_train,
                policy,
                gamma=gamma,
                max_steps=1000
            )
            
            # Update policy (only if we have data)
            if len(states) > 0:
                update_stats = trainer.update_policy(
                    states=states,
                    actions=actions,
                    old_log_probs=old_log_probs,
                    advantages=advantages,
                    returns=returns
                )
            else:
                update_stats = {
                    "policy_loss": 0.0,
                    "value_loss": 0.0,
                    "entropy": 0.0,
                    "total_loss": 0.0,
                    "n_updates": 0
                }
            
            # Evaluate on test environment
            test_reward = evaluate(env_test, policy, n_episodes=5, deterministic=True)
            
            # Update statistics
            training_stats["episode_rewards"].append(train_reward)
            training_stats["test_rewards"].append(test_reward)
            training_stats["policy_losses"].append(update_stats["policy_loss"])
            training_stats["value_losses"].append(update_stats["value_loss"])
            training_stats["entropies"].append(update_stats["entropy"])
            training_stats["episodes_trained"] = episode
            
            # Track best model
            if test_reward > training_stats["best_test_reward"]:
                training_stats["best_test_reward"] = test_reward
                torch.save(policy.state_dict(), model_save_path)
                logger.info(f"✅ New best model saved (test reward: {test_reward:.2f})")
            
            # Log metrics
            if episode % log_interval == 0:
                avg_train_reward = np.mean(training_stats["episode_rewards"][-log_interval:])
                avg_test_reward = np.mean(training_stats["test_rewards"][-log_interval:])
                avg_pol_loss = np.mean(training_stats["policy_losses"][-log_interval:])
                avg_val_loss = np.mean(training_stats["value_losses"][-log_interval:])
                avg_entropy = np.mean(training_stats["entropies"][-log_interval:])
                
                logger.info(
                    f"Episode {episode}/{max_episodes} | "
                    f"Train Reward: {avg_train_reward:.2f} | "
                    f"Test Reward: {avg_test_reward:.2f} | "
                    f"Pol Loss: {avg_pol_loss:.4f} | "
                    f"Val Loss: {avg_val_loss:.4f} | "
                    f"Entropy: {avg_entropy:.4f}"
                )
                
                if use_mlflow:
                    try:
                        import mlflow
                        mlflow.log_metrics({
                            "train_reward": avg_train_reward,
                            "test_reward": avg_test_reward,
                            "policy_loss": avg_pol_loss,
                            "value_loss": avg_val_loss,
                            "entropy": avg_entropy
                        }, step=episode)
                    except Exception as e:
                        logger.warning(f"Failed to log to MLflow: {e}")
            
            # Save model periodically
            if episode % save_interval == 0:
                checkpoint_path = f"{model_save_path}.checkpoint_{episode}"
                torch.save({
                    "episode": episode,
                    "policy_state_dict": policy.state_dict(),
                    "optimizer_state_dict": trainer.optimizer.state_dict(),
                    "training_stats": training_stats
                }, checkpoint_path)
                logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            # Early stopping
            if test_reward >= threshold:
                logger.info(f"✅ Early stopping: Test reward {test_reward:.2f} >= threshold {threshold}")
                break
        
        except Exception as e:
            logger.error(f"Error in training episode {episode}: {e}")
            continue
    
    # Final evaluation
    logger.info("Running final evaluation...")
    final_test_reward = evaluate(env_test, policy, n_episodes=10, deterministic=True)
    training_stats["final_test_reward"] = final_test_reward
    
    logger.info(f"Training completed!")
    logger.info(f"Episodes trained: {training_stats['episodes_trained']}")
    logger.info(f"Best test reward: {training_stats['best_test_reward']:.2f}")
    logger.info(f"Final test reward: {final_test_reward:.2f}")
    
    if use_mlflow and mlflow_run:
        try:
            import mlflow
            mlflow.log_metric("final_test_reward", final_test_reward)
            mlflow.log_metric("best_test_reward", training_stats["best_test_reward"])
            mlflow.log_artifact(model_save_path, "model")
            mlflow.end_run()
        except Exception as e:
            logger.warning(f"Failed to finalize MLflow run: {e}")
    
    return {
        "policy": policy,
        "trainer": trainer,
        "training_stats": training_stats,
        "model_path": model_save_path
    }

