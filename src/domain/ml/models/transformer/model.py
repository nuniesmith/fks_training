"""
Transformer Model for Time-Series Forecasting

Uses self-attention mechanisms to capture long-range dependencies
in financial time-series data.
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
class TransformerConfig:
    """Configuration for Transformer model"""

    sequence_length: int = 60
    input_features: int = 5  # OHLCV
    d_model: int = 128  # Model dimension
    nhead: int = 8  # Number of attention heads
    num_layers: int = 4  # Number of transformer layers
    dim_feedforward: int = 512  # Feedforward dimension
    dropout: float = 0.1
    learning_rate: float = 0.0001
    batch_size: int = 32
    epochs: int = 100
    device: str = "cpu"


class TimeSeriesDataset(Dataset):
    """Dataset for time-series data"""

    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx]


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding"""
        return x + self.pe[: x.size(0), :]


class TransformerModel(nn.Module):
    """
    Transformer model for time-series forecasting.

    Uses self-attention to capture long-range dependencies.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.input_projection = nn.Linear(config.input_features, config.d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(config.d_model, config.sequence_length)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=False,  # (seq_len, batch, features)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers
        )

        # Output projection
        self.output_projection = nn.Linear(config.d_model, 1)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, sequence_length, features)

        Returns:
            Predictions of shape (batch, 1)
        """
        # Project input to model dimension
        x = self.input_projection(x)  # (batch, seq_len, d_model)

        # Transpose for transformer: (seq_len, batch, d_model)
        x = x.transpose(0, 1)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Apply transformer encoder
        x = self.transformer_encoder(x)

        # Take last timestep
        x = x[-1, :, :]  # (batch, d_model)

        # Apply dropout and output projection
        x = self.dropout(x)
        x = self.output_projection(x)  # (batch, 1)

        return x


class TransformerTrainer:
    """Trainer for Transformer models"""

    def __init__(self, config: TransformerConfig):
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        self.model = TransformerModel(config).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.criterion = nn.MSELoss()

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        run_name: str = "Transformer_Training",
    ) -> Dict[str, List[float]]:
        """Train the Transformer model."""
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

