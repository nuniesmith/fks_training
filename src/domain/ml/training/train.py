"""
Training script for LSTM models.

This script can be run via MLflow Projects:
    mlflow run . -e train --experiment-name lstm_forecasting
"""

import argparse
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import mlflow
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from domain.ml.features.indicators import TechnicalIndicators
from domain.ml.features.preprocessing import FeaturePreprocessor
from domain.ml.models.lstm.model import LSTMModel, LSTMModelConfig, LSTMTrainer, TimeSeriesDataset


def load_data(
    symbol: str,
    data_path: Optional[str] = None,
    interval: str = "1h",
    days_back: int = 30,
    data_client: Optional[Any] = None,
) -> pd.DataFrame:
    """
    Load market data for training.

    Args:
        symbol: Trading symbol (e.g., "BTC/USDT")
        data_path: Optional path to CSV file with OHLCV data
        interval: Time interval for data (default: "1h")
        days_back: Number of days of historical data (default: 30)
        data_client: Optional DataClient instance (will create if not provided)

    Returns:
        DataFrame with OHLCV data (columns: timestamp, open, high, low, close, volume)
    """
    if data_path:
        # Load from CSV file
        df = pd.read_csv(data_path)
        
        # Normalize timestamp column
        if "timestamp" not in df.columns:
            if "ts" in df.columns:
                df = df.rename(columns={"ts": "timestamp"})
            elif "time" in df.columns:
                df = df.rename(columns={"time": "timestamp"})
            elif "date" in df.columns:
                df = df.rename(columns={"date": "timestamp"})
        
        if "timestamp" in df.columns:
            if df["timestamp"].dtype == "int64":
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            elif df["timestamp"].dtype == "object":
                df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        # Load from fks_data service
        if data_client is None:
            from infrastructure.data_client import DataClient
            
            data_client = DataClient()
        
        # Calculate time range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days_back)
        
        print(f"Fetching data from fks_data service: {symbol} ({interval})")
        df = data_client.fetch_ohlcv(
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time,
        )
        
        print(f"Fetched {len(df)} records")

    # Ensure required columns exist
    required_cols = ["open", "high", "low", "close", "volume"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    # Ensure timestamp column exists
    if "timestamp" not in df.columns:
        # Create a dummy timestamp if missing
        df["timestamp"] = pd.date_range(start="2024-01-01", periods=len(df), freq="1H")

    return df


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train LSTM model for price forecasting")
    parser.add_argument("--model-name", type=str, default="lstm_price_forecast")
    parser.add_argument("--symbol", type=str, default="BTC/USDT")
    parser.add_argument("--data-path", type=str, help="Path to CSV file with OHLCV data")
    parser.add_argument("--sequence-length", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden-units", type=int, default=50)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--interval", type=str, default="1h", help="Data interval (e.g., 1m, 5m, 1h, 1d)")
    parser.add_argument("--days-back", type=int, default=30, help="Number of days of historical data")

    args = parser.parse_args()

    # Load and prepare data
    print(f"Loading data for {args.symbol}...")
    df = load_data(
        args.symbol,
        args.data_path if args.data_path else None,
        interval=args.interval,
        days_back=args.days_back,
    )

    # Create technical indicators
    print("Creating technical indicators...")
    df = TechnicalIndicators.create_all_indicators(df)

    # Prepare data
    print("Preparing data...")
    preprocessor = FeaturePreprocessor(
        scaler_type="minmax",
        target_column="close",
    )

    (train_sequences, train_targets), (val_sequences, val_targets), _ = (
        preprocessor.prepare_data(
            df,
            sequence_length=args.sequence_length,
            prediction_horizon=1,
            train_split=0.8,
        )
    )

    print(f"Training sequences: {len(train_sequences)}")
    print(f"Validation sequences: {len(val_sequences)}")
    print(f"Feature dimensions: {train_sequences.shape[2]}")

    # Create data loaders
    train_dataset = TimeSeriesDataset(train_sequences, train_targets)
    val_dataset = TimeSeriesDataset(val_sequences, val_targets)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model config
    config = LSTMModelConfig(
        sequence_length=args.sequence_length,
        input_features=train_sequences.shape[2],
        hidden_units=args.hidden_units,
        num_layers=args.num_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=args.device,
    )

    # Train model
    print("Starting training...")
    trainer = LSTMTrainer(config)
    history = trainer.train(
        train_loader,
        val_loader,
        run_name=f"{args.model_name}_{args.symbol}",
    )

    # Log additional metrics
    final_train_loss = history["train_loss"][-1]
    final_val_loss = history["val_loss"][-1] if history["val_loss"] else None

    mlflow.log_metric("final_train_loss", final_train_loss)
    if final_val_loss:
        mlflow.log_metric("final_val_loss", final_val_loss)
        # Calculate best validation loss
        best_val_loss = min(history["val_loss"])
        best_epoch = history["val_loss"].index(best_val_loss)
        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.log_param("best_epoch", best_epoch)

    # Log data statistics
    mlflow.log_param("n_train_samples", len(train_sequences))
    mlflow.log_param("n_val_samples", len(val_sequences))
    mlflow.log_param("n_features", train_sequences.shape[2])
    mlflow.log_param("sequence_length", args.sequence_length)

    # Log model performance summary
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write(f"Training Summary\n")
        f.write(f"================\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Symbol: {args.symbol}\n")
        f.write(f"Final Train Loss: {final_train_loss:.6f}\n")
        if final_val_loss:
            f.write(f"Final Val Loss: {final_val_loss:.6f}\n")
            f.write(f"Best Val Loss: {best_val_loss:.6f} (Epoch {best_epoch})\n")
        f.write(f"Training Samples: {len(train_sequences)}\n")
        f.write(f"Validation Samples: {len(val_sequences)}\n")
        f.write(f"Features: {train_sequences.shape[2]}\n")
        f.flush()
        mlflow.log_artifact(f.name, "summary")
        os.unlink(f.name)

    print("Training completed!")
    print(f"Final train loss: {final_train_loss:.4f}")
    if final_val_loss:
        print(f"Final validation loss: {final_val_loss:.4f}")
        print(f"Best validation loss: {best_val_loss:.4f} (Epoch {best_epoch})")


if __name__ == "__main__":
    main()

