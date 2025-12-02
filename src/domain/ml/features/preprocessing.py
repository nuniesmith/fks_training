"""
Feature Preprocessing for ML Models

Handles data normalization, scaling, and sequence creation for time-series models.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import List, Optional, Tuple


class FeaturePreprocessor:
    """
    Preprocess features for ML models.

    Handles:
    - Scaling/normalization
    - Sequence creation for time-series models
    - Feature selection
    - Missing value handling
    """

    def __init__(
        self,
        scaler_type: str = "minmax",
        feature_columns: Optional[List[str]] = None,
        target_column: str = "close",
    ):
        """
        Initialize preprocessor.

        Args:
            scaler_type: Type of scaler ('minmax' or 'standard')
            feature_columns: List of feature column names (None = use all numeric)
            target_column: Name of target column for prediction
        """
        self.scaler_type = scaler_type
        self.feature_columns = feature_columns
        self.target_column = target_column

        if scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        elif scaler_type == "standard":
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")

        self.target_scaler = MinMaxScaler()
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> "FeaturePreprocessor":
        """
        Fit scalers on training data.

        Args:
            df: Training DataFrame

        Returns:
            Self for method chaining
        """
        # Determine feature columns
        if self.feature_columns is None:
            # Use all numeric columns except target
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            self.feature_columns = [col for col in numeric_cols if col != self.target_column]

        # Fit feature scaler
        feature_data = df[self.feature_columns].fillna(method="ffill").fillna(0)
        self.scaler.fit(feature_data)

        # Fit target scaler
        if self.target_column in df.columns:
            target_data = df[[self.target_column]].fillna(method="ffill").fillna(0)
            self.target_scaler.fit(target_data)

        self.is_fitted = True
        return self

    def transform(
        self, df: pd.DataFrame, include_target: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transform data using fitted scalers.

        Args:
            df: DataFrame to transform
            include_target: Whether to include target column

        Returns:
            Tuple of (features, targets) as numpy arrays
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        # Handle missing values
        feature_data = df[self.feature_columns].fillna(method="ffill").fillna(0)
        features = self.scaler.transform(feature_data)

        targets = None
        if include_target and self.target_column in df.columns:
            target_data = df[[self.target_column]].fillna(method="ffill").fillna(0)
            targets = self.target_scaler.transform(target_data).flatten()

        return features, targets

    def inverse_transform_target(self, scaled_targets: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled targets back to original scale.

        Args:
            scaled_targets: Scaled target values

        Returns:
            Original scale target values
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before inverse_transform")

        return self.target_scaler.inverse_transform(scaled_targets.reshape(-1, 1)).flatten()

    def create_sequences(
        self,
        features: np.ndarray,
        targets: Optional[np.ndarray] = None,
        sequence_length: int = 60,
        prediction_horizon: int = 1,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create sequences for time-series models.

        Args:
            features: Feature array of shape (n_samples, n_features)
            targets: Target array of shape (n_samples,)
            sequence_length: Length of input sequences
            prediction_horizon: How many steps ahead to predict (default: 1)

        Returns:
            Tuple of (sequences, sequence_targets)
            - sequences: Shape (n_sequences, sequence_length, n_features)
            - sequence_targets: Shape (n_sequences,)
        """
        n_samples = len(features)
        n_features = features.shape[1]

        # Calculate number of sequences
        n_sequences = n_samples - sequence_length - prediction_horizon + 1

        if n_sequences <= 0:
            raise ValueError(
                f"Not enough data for sequences. Need at least {sequence_length + prediction_horizon} samples"
            )

        # Create sequences
        sequences = np.zeros((n_sequences, sequence_length, n_features))
        sequence_targets = None

        if targets is not None:
            sequence_targets = np.zeros(n_sequences)

        for i in range(n_sequences):
            sequences[i] = features[i : i + sequence_length]
            if targets is not None:
                # Target is the value at sequence_length + prediction_horizon - 1 steps ahead
                sequence_targets[i] = targets[i + sequence_length + prediction_horizon - 1]

        return sequences, sequence_targets

    def prepare_data(
        self,
        df: pd.DataFrame,
        sequence_length: int = 60,
        prediction_horizon: int = 1,
        train_split: float = 0.8,
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
        "FeaturePreprocessor",
    ]:
        """
        Complete data preparation pipeline.

        Args:
            df: Input DataFrame with features and target
            sequence_length: Length of input sequences
            prediction_horizon: Steps ahead to predict
            train_split: Fraction of data for training (default: 0.8)

        Returns:
            Tuple of ((train_sequences, train_targets), (val_sequences, val_targets), preprocessor)
        """
        # Fit on full dataset
        self.fit(df)

        # Split data
        split_idx = int(len(df) * train_split)
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]

        # Transform
        train_features, train_targets = self.transform(train_df, include_target=True)
        val_features, val_targets = self.transform(val_df, include_target=True)

        # Create sequences
        train_sequences, train_seq_targets = self.create_sequences(
            train_features, train_targets, sequence_length, prediction_horizon
        )
        val_sequences, val_seq_targets = self.create_sequences(
            val_features, val_targets, sequence_length, prediction_horizon
        )

        return (
            (train_sequences, train_seq_targets),
            (val_sequences, val_seq_targets),
            self,
        )

