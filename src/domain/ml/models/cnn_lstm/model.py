"""
CNN-LSTM Hybrid Model for Multi-Asset Correlation Analysis

Combines CNN for spatial feature extraction (multi-asset correlations)
with LSTM for temporal sequence modeling.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


@dataclass
class CNNLSTMConfig:
    """Configuration for CNN-LSTM model"""

    sequence_length: int = 60
    num_assets: int = 5  # Number of correlated assets
    input_features: int = 5  # OHLCV per asset
    cnn_filters: int = 64
    cnn_kernel_size: int = 3
    lstm_hidden_units: int = 50
    lstm_num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    device: str = "cpu"


class MultiAssetDataset(Dataset):
    """Dataset for multi-asset time-series data"""

    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        """
        Args:
            sequences: Input sequences of shape (n_samples, sequence_length, num_assets, n_features)
            targets: Target values of shape (n_samples,)
        """
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx]


class CNNLSTMModel(nn.Module):
    """
    CNN-LSTM hybrid model for multi-asset correlation analysis.

    Architecture:
    1. CNN layers extract spatial features (correlations between assets)
    2. LSTM layers model temporal dependencies
    3. Fully connected layer produces predictions
    """

    def __init__(self, config: CNNLSTMConfig):
        super().__init__()
        self.config = config

        # CNN for spatial feature extraction
        # Input: (batch, sequence_length, num_assets, features)
        # We'll apply CNN along the asset dimension
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=config.input_features,
                out_channels=config.cnn_filters,
                kernel_size=(1, config.cnn_kernel_size),
                padding=(0, config.cnn_kernel_size // 2),
            ),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Conv2d(
                in_channels=config.cnn_filters,
                out_channels=config.cnn_filters,
                kernel_size=(1, config.cnn_kernel_size),
                padding=(0, config.cnn_kernel_size // 2),
            ),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        # Calculate CNN output size
        # After CNN: (batch, cnn_filters, sequence_length, num_assets)
        # Reshape for LSTM: (batch, sequence_length, cnn_filters * num_assets)
        lstm_input_size = config.cnn_filters * config.num_assets

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=config.lstm_hidden_units,
            num_layers=config.lstm_num_layers,
            batch_first=True,
            dropout=config.dropout if config.lstm_num_layers > 1 else 0,
        )

        # Output layer
        self.fc = nn.Linear(config.lstm_hidden_units, 1)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, sequence_length, num_assets, features)

        Returns:
            Predictions of shape (batch, 1)
        """
        batch_size, seq_len, num_assets, features = x.shape

        # Reshape for CNN: (batch, features, sequence_length, num_assets)
        x = x.permute(0, 3, 1, 2)

        # Apply CNN
        cnn_out = self.cnn(x)  # (batch, cnn_filters, sequence_length, num_assets)

        # Reshape for LSTM: (batch, sequence_length, cnn_filters * num_assets)
        cnn_out = cnn_out.permute(0, 2, 1, 3)
        cnn_out = cnn_out.reshape(batch_size, seq_len, -1)

        # Apply LSTM
        lstm_out, _ = self.lstm(cnn_out)

        # Take last output
        last_output = lstm_out[:, -1, :]

        # Apply dropout and fully connected layer
        output = self.dropout(last_output)
        output = self.fc(output)

        return output


class CNNLSTMTrainer:
    """Trainer for CNN-LSTM models"""

    def __init__(self, config: CNNLSTMConfig):
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        self.model = CNNLSTMModel(config).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.criterion = nn.MSELoss()

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        run_name: str = "CNN_LSTM_Training",
    ) -> Dict[str, List[float]]:
        """Train the CNN-LSTM model."""
        history = {"train_loss": [], "val_loss": []}

        with mlflow.start_run(run_name=run_name):
            # Log configuration
            mlflow.log_params(self.config.__dict__)

            for epoch in range(self.config.epochs):
                # Training
                train_loss = self._train_epoch(train_loader)
                history["train_loss"].append(train_loss)

                # Validation
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
                    logger.info(
                        f"Epoch {epoch+1}/{self.config.epochs} - "
                        f"Train Loss: {train_loss:.4f}"
                        + (f", Val Loss: {val_loss:.4f}" if val_loss else "")
                    )

            # Log model
            mlflow.pytorch.log_model(self.model, "model")

        return history

    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for sequences, targets in train_loader:
            sequences = sequences.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(sequences)
            loss = self.criterion(predictions, targets.unsqueeze(1))

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

