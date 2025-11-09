"""
LSTM Model for Price Forecasting

Implements LSTM-based time-series models for predicting asset prices.
Supports OHLCV data with technical indicators as features.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ...features.indicators import TechnicalIndicators
from ...features.preprocessing import FeaturePreprocessor


@dataclass
class LSTMModelConfig:
    """Configuration for LSTM model"""

    sequence_length: int = 60
    input_features: int = 5  # OHLCV by default
    hidden_units: int = 50
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    device: str = "cpu"


class TimeSeriesDataset(Dataset):
    """Dataset for time-series data"""

    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        """
        Args:
            sequences: Input sequences of shape (n_samples, sequence_length, n_features)
            targets: Target values of shape (n_samples,)
        """
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx]


class LSTMModel(nn.Module):
    """
    LSTM model for price forecasting

    Architecture:
    - LSTM layers for sequence modeling
    - Dropout for regularization
    - Fully connected layer for output
    """

    def __init__(self, config: LSTMModelConfig):
        super().__init__()
        self.config = config

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.input_features,
            hidden_size=config.hidden_units,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True,
        )

        # Output layer
        self.fc = nn.Linear(config.hidden_units, 1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_features)

        Returns:
            Predictions of shape (batch_size, 1)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # Take the last output
        last_output = lstm_out[:, -1, :]

        # Fully connected layer
        output = self.fc(last_output)

        return output

    def predict(self, sequences: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data

        Args:
            sequences: Input sequences of shape (n_samples, sequence_length, n_features)

        Returns:
            Predictions of shape (n_samples,)
        """
        self.eval()
        device = next(self.parameters()).device

        with torch.no_grad():
            sequences_tensor = torch.FloatTensor(sequences).to(device)
            predictions = self.forward(sequences_tensor)
            return predictions.cpu().numpy().flatten()


class LSTMTrainer:
    """Trainer for LSTM models with MLflow integration"""

    def __init__(self, config: LSTMModelConfig):
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        self.model = LSTMModel(config).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate
        )
        self.criterion = nn.MSELoss()

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        run_name: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the model with MLflow tracking

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            run_name: MLflow run name

        Returns:
            Dictionary with training history
        """
        history = {"train_loss": [], "val_loss": []}

        with mlflow.start_run(run_name=run_name):
            # Log hyperparameters
            mlflow.log_params(
                {
                    "sequence_length": self.config.sequence_length,
                    "hidden_units": self.config.hidden_units,
                    "num_layers": self.config.num_layers,
                    "dropout": self.config.dropout,
                    "learning_rate": self.config.learning_rate,
                    "batch_size": self.config.batch_size,
                    "epochs": self.config.epochs,
                }
            )

            for epoch in range(self.config.epochs):
                # Training phase
                train_loss = self._train_epoch(train_loader)
                history["train_loss"].append(train_loss)

                # Validation phase
                if val_loader:
                    val_loss = self._validate_epoch(val_loader)
                    history["val_loss"].append(val_loss)
                else:
                    val_loss = None

                # Log metrics
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                if val_loss is not None:
                    mlflow.log_metric("val_loss", val_loss, step=epoch)

                if (epoch + 1) % 10 == 0:
                    print(
                        f"Epoch {epoch+1}/{self.config.epochs} - "
                        f"Train Loss: {train_loss:.4f}"
                        + (f", Val Loss: {val_loss:.4f}" if val_loss else "")
                    )

            # Log model
            mlflow.pytorch.log_model(self.model, "model")
            
            # Log model artifacts
            import tempfile
            import os
            
            # Log model architecture summary
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
                # Count parameters
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                
                f.write(f"Model Architecture Summary\n")
                f.write(f"==========================\n")
                f.write(f"Total Parameters: {total_params:,}\n")
                f.write(f"Trainable Parameters: {trainable_params:,}\n")
                f.write(f"Hidden Units: {self.config.hidden_units}\n")
                f.write(f"Number of Layers: {self.config.num_layers}\n")
                f.write(f"Input Features: {self.config.input_features}\n")
                f.write(f"Sequence Length: {self.config.sequence_length}\n")
                f.write(f"Dropout: {self.config.dropout}\n")
                f.write(f"\nModel:\n{self.model}\n")
                f.flush()
                mlflow.log_artifact(f.name, "model_info")
                os.unlink(f.name)
            
            # Log training history plot if matplotlib available
            try:
                import matplotlib
                matplotlib.use("Agg")  # Non-interactive backend
                import matplotlib.pyplot as plt
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(history["train_loss"], label="Train Loss")
                if history["val_loss"]:
                    ax.plot(history["val_loss"], label="Validation Loss")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.set_title("Training History")
                ax.legend()
                ax.grid(True)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
                    fig.savefig(f.name, dpi=150, bbox_inches="tight")
                    mlflow.log_artifact(f.name, "plots")
                    os.unlink(f.name)
                plt.close(fig)
            except ImportError:
                pass  # matplotlib not available, skip plotting

        return history

    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for sequences, targets in train_loader:
            sequences = sequences.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(sequences)
            loss = self.criterion(predictions, targets.unsqueeze(1))

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)

                predictions = self.model(sequences)
                loss = self.criterion(predictions, targets.unsqueeze(1))

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def save_model(self, path: str):
        """Save model to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": self.config,
            },
            path,
        )

    @classmethod
    def load_model(cls, path: str, device: str = "cpu") -> LSTMModel:
        """Load model from disk"""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint["config"]
        model = LSTMModel(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

