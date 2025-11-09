"""
Unit tests for ML models

Tests LSTM, CNN-LSTM, and Transformer models.
"""

import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader

from src.domain.ml.models.lstm.model import (
    LSTMModel,
    LSTMModelConfig,
    LSTMTrainer,
    TimeSeriesDataset,
)
from src.domain.ml.models.cnn_lstm.model import (
    CNNLSTMModel,
    CNNLSTMConfig,
    CNNLSTMTrainer,
    MultiAssetDataset,
)
from src.domain.ml.models.transformer.model import (
    TransformerModel,
    TransformerConfig,
    TransformerTrainer,
    TimeSeriesDataset as TransformerDataset,
)


class TestLSTMModel:
    """Tests for LSTM model"""

    def test_lstm_model_creation(self):
        """Test LSTM model can be created"""
        config = LSTMModelConfig(
            sequence_length=60,
            input_features=5,
            hidden_units=50,
            num_layers=2,
        )
        model = LSTMModel(config)
        assert model is not None

    def test_lstm_forward_pass(self):
        """Test LSTM forward pass"""
        config = LSTMModelConfig(sequence_length=60, input_features=5)
        model = LSTMModel(config)
        model.eval()

        batch_size = 4
        x = torch.randn(batch_size, config.sequence_length, config.input_features)
        output = model(x)

        assert output.shape == (batch_size, 1)

    def test_lstm_training_step(self):
        """Test LSTM training step"""
        config = LSTMModelConfig(
            sequence_length=10,
            input_features=5,
            hidden_units=20,
            epochs=1,
            batch_size=2,
        )
        trainer = LSTMTrainer(config)

        # Create dummy data
        n_samples = 10
        sequences = np.random.randn(n_samples, config.sequence_length, config.input_features)
        targets = np.random.randn(n_samples)

        dataset = TimeSeriesDataset(sequences, targets)
        loader = DataLoader(dataset, batch_size=config.batch_size)

        # Train one epoch
        history = trainer.train(loader, run_name="test_lstm")
        assert "train_loss" in history
        assert len(history["train_loss"]) == 1


class TestCNNLSTMModel:
    """Tests for CNN-LSTM model"""

    def test_cnn_lstm_model_creation(self):
        """Test CNN-LSTM model can be created"""
        config = CNNLSTMConfig(
            sequence_length=60,
            num_assets=5,
            input_features=5,
        )
        model = CNNLSTMModel(config)
        assert model is not None

    def test_cnn_lstm_forward_pass(self):
        """Test CNN-LSTM forward pass"""
        config = CNNLSTMConfig(sequence_length=10, num_assets=3, input_features=5)
        model = CNNLSTMModel(config)
        model.eval()

        batch_size = 2
        x = torch.randn(
            batch_size, config.sequence_length, config.num_assets, config.input_features
        )
        output = model(x)

        assert output.shape == (batch_size, 1)


class TestTransformerModel:
    """Tests for Transformer model"""

    def test_transformer_model_creation(self):
        """Test Transformer model can be created"""
        config = TransformerConfig(
            sequence_length=60,
            input_features=5,
            d_model=128,
            nhead=8,
        )
        model = TransformerModel(config)
        assert model is not None

    def test_transformer_forward_pass(self):
        """Test Transformer forward pass"""
        config = TransformerConfig(sequence_length=10, input_features=5, d_model=64)
        model = TransformerModel(config)
        model.eval()

        batch_size = 2
        x = torch.randn(batch_size, config.sequence_length, config.input_features)
        output = model(x)

        assert output.shape == (batch_size, 1)


class TestDataLoaders:
    """Tests for data loaders"""

    def test_time_series_dataset(self):
        """Test TimeSeriesDataset"""
        n_samples = 10
        sequence_length = 5
        n_features = 3

        sequences = np.random.randn(n_samples, sequence_length, n_features)
        targets = np.random.randn(n_samples)

        dataset = TimeSeriesDataset(sequences, targets)
        assert len(dataset) == n_samples

        sample_seq, sample_target = dataset[0]
        assert sample_seq.shape == (sequence_length, n_features)
        assert sample_target.shape == ()

    def test_multi_asset_dataset(self):
        """Test MultiAssetDataset"""
        n_samples = 10
        sequence_length = 5
        num_assets = 3
        n_features = 5

        sequences = np.random.randn(n_samples, sequence_length, num_assets, n_features)
        targets = np.random.randn(n_samples)

        dataset = MultiAssetDataset(sequences, targets)
        assert len(dataset) == n_samples

        sample_seq, sample_target = dataset[0]
        assert sample_seq.shape == (sequence_length, num_assets, n_features)
        assert sample_target.shape == ()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

