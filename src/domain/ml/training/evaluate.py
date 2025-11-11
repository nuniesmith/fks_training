"""
Model Evaluation Script

Evaluates trained models on test data and generates performance metrics.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from domain.ml.features.indicators import TechnicalIndicators
from domain.ml.features.preprocessing import FeaturePreprocessor
from domain.ml.models.lstm.model import LSTMModel, LSTMModelConfig, LSTMTrainer, TimeSeriesDataset
from torch.utils.data import DataLoader


def load_test_data(data_path: str) -> pd.DataFrame:
    """
    Load test data for evaluation.

    Args:
        data_path: Path to CSV file with test data

    Returns:
        DataFrame with OHLCV data
    """
    df = pd.read_csv(data_path)
    
    # Ensure required columns exist
    required_cols = ["open", "high", "low", "close", "volume"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    
    return df


def calculate_trading_metrics(
    predictions: np.ndarray,
    actuals: np.ndarray,
    prices: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate trading-specific metrics.

    Args:
        predictions: Model predictions
        actuals: Actual values
        prices: Historical prices for return calculation

    Returns:
        Dictionary of trading metrics
    """
    metrics = {}
    
    # Direction accuracy (percentage of correct direction predictions)
    if len(predictions) > 1 and len(actuals) > 1:
        pred_direction = np.diff(predictions) > 0
        actual_direction = np.diff(actuals) > 0
        direction_accuracy = (pred_direction == actual_direction).mean()
        metrics["direction_accuracy"] = float(direction_accuracy)
    
    # Calculate returns if prices available
    if len(prices) > 1:
        # Simple strategy: buy when prediction > current price, sell otherwise
        returns = []
        for i in range(1, len(predictions)):
            if predictions[i] > prices[i-1]:
                # Buy signal
                if i < len(prices):
                    returns.append((prices[i] - prices[i-1]) / prices[i-1])
            else:
                # Sell signal (or hold)
                returns.append(0.0)
        
        if returns:
            total_return = sum(returns)
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)  # Annualized
            max_drawdown = calculate_max_drawdown(returns)
            
            metrics["total_return"] = float(total_return)
            metrics["sharpe_ratio"] = float(sharpe_ratio)
            metrics["max_drawdown"] = float(max_drawdown)
    
    return metrics


def calculate_max_drawdown(returns: list) -> float:
    """Calculate maximum drawdown from returns."""
    cumulative = [1 + r for r in returns]
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return abs(min(drawdown)) if len(drawdown) > 0 else 0.0


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate ML model on test data")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model file or MLflow model URI")
    parser.add_argument("--test-data-path", type=str, required=True, help="Path to test data CSV")
    parser.add_argument("--sequence-length", type=int, default=60, help="Sequence length used in training")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu/cuda)")
    parser.add_argument("--tracking-uri", type=str, help="MLflow tracking URI")
    parser.add_argument("--run-id", type=str, help="MLflow run ID to log metrics to")

    args = parser.parse_args()

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)

    # Load test data
    print(f"Loading test data from {args.test_data_path}...")
    test_df = load_test_data(args.test_data_path)

    # Create technical indicators
    print("Creating technical indicators...")
    test_df = TechnicalIndicators.create_all_indicators(test_df)

    # Prepare data
    print("Preparing data...")
    preprocessor = FeaturePreprocessor(
        scaler_type="minmax",
        target_column="close",
    )
    
    # Fit on test data (in production, would use training scaler)
    preprocessor.fit(test_df)
    
    # Transform test data
    test_features, test_targets = preprocessor.transform(test_df, include_target=True)
    
    # Create sequences
    test_sequences, test_seq_targets = preprocessor.create_sequences(
        test_features, test_targets, sequence_length=args.sequence_length, prediction_horizon=1
    )

    print(f"Test sequences: {len(test_sequences)}")

    # Load model
    print(f"Loading model from {args.model_path}...")
    if args.model_path.startswith("models:/") or args.model_path.startswith("runs:/"):
        # Load from MLflow
        model = mlflow.pyfunc.load_model(args.model_path)
        # For PyTorch models, need to extract the actual model
        if hasattr(model, "unwrap_python_model"):
            model = model.unwrap_python_model()
    else:
        # Load from file
        config = LSTMModelConfig(
            sequence_length=args.sequence_length,
            input_features=test_sequences.shape[2],
        )
        model = LSTMTrainer.load_model(args.model_path, device=args.device)

    # Create data loader
    test_dataset = TimeSeriesDataset(test_sequences, test_seq_targets)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Evaluate model
    print("Evaluating model...")
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for sequences, targets in test_loader:
            sequences = sequences.to(args.device)
            targets = targets.to(args.device)

            predictions = model(sequences)
            all_predictions.extend(predictions.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())

    # Convert to numpy arrays
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)

    # Inverse transform to original scale
    predictions_original = preprocessor.inverse_transform_target(predictions)
    targets_original = preprocessor.inverse_transform_target(targets)

    # Calculate regression metrics
    mse = mean_squared_error(targets_original, predictions_original)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets_original, predictions_original)
    r2 = r2_score(targets_original, predictions_original)

    # Calculate trading metrics
    trading_metrics = calculate_trading_metrics(
        predictions_original, targets_original, test_df["close"].values[-len(predictions_original):]
    )

    # Combine all metrics
    all_metrics = {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2_score": float(r2),
        **trading_metrics,
    }

    # Log to MLflow if run_id provided
    if args.run_id:
        with mlflow.start_run(run_id=args.run_id):
            mlflow.log_metrics(all_metrics)
            print(f"Metrics logged to MLflow run: {args.run_id}")
    else:
        # Create new run
        with mlflow.start_run(run_name="model_evaluation"):
            mlflow.log_metrics(all_metrics)
            print(f"Metrics logged to new MLflow run: {mlflow.active_run().info.run_id}")

    # Print results
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"MSE:  {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"R²:   {r2:.4f}")
    
    if trading_metrics:
        print("\nTrading Metrics:")
        for key, value in trading_metrics.items():
            print(f"  {key}: {value:.4f}")
    
    print("="*50)

    return all_metrics


if __name__ == "__main__":
    main()

