from abc import abstractmethod
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from framework.base.pipeline import BasePipeline
from sklearn.model_selection import TimeSeriesSplit


class SupervisedPipeline(BasePipeline):
    """Pipeline for supervised learning models"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.feature_pipeline = None
        self.model = None

    async def train(self) -> Any:
        """Execute training pipeline"""
        # Load data
        data = await self._load_data()

        # Feature engineering
        features, labels = await self._prepare_features(data)

        # Split data
        train_data, val_data = await self._split_data(features, labels)

        # Hyperparameter tuning if enabled
        if self.config.get("tune_hyperparameters", False):
            best_params = await self._tune_hyperparameters(train_data)
            self.config["model_params"].update(best_params)

        # Train model
        self.model = await self._train_model(train_data)

        # Validate
        val_metrics = await self._validate_model(self.model, val_data)

        return self.model

    async def _prepare_features(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for training"""
        # Technical indicators
        data = await self._add_technical_indicators(data)

        # Market microstructure features
        data = await self._add_microstructure_features(data)

        # Sentiment features if available
        if self.config.get("use_sentiment", False):
            data = await self._add_sentiment_features(data)

        # Create target variable
        data["target"] = self._create_target(data)

        # Select features
        feature_cols = self.config["features"]
        features = data[feature_cols]
        labels = data["target"]

        return features, labels

    async def _tune_hyperparameters(self, train_data: Tuple) -> Dict[str, Any]:
        """Hyperparameter optimization using Optuna"""
        import optuna

        def objective(trial):
            # Suggest hyperparameters
            params = {
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            }

            # Train with suggested params
            model = self._create_model(params)

            # Use time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            scores = []

            for train_idx, val_idx in tscv.split(train_data[0]):
                X_train, X_val = (
                    train_data[0].iloc[train_idx],
                    train_data[0].iloc[val_idx],
                )
                y_train, y_val = (
                    train_data[1].iloc[train_idx],
                    train_data[1].iloc[val_idx],
                )

                model.fit(X_train, y_train)
                score = model.score(X_val, y_val)
                scores.append(score)

            return np.mean(scores)

        # Create study
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.config.get("n_trials", 100))

        return study.best_params
