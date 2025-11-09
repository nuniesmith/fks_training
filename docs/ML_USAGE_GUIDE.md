# ML Usage Guide for FKS Trading Platform

This guide provides comprehensive documentation for using the ML/AI features in the FKS Trading Platform.

## Table of Contents

1. [Overview](#overview)
2. [Training Models](#training-models)
3. [Making Predictions](#making-predictions)
4. [Model Management](#model-management)
5. [Backtesting with ML](#backtesting-with-ml)
6. [Sentiment Integration](#sentiment-integration)
7. [Advanced Models](#advanced-models)

## Overview

The FKS Trading Platform provides a complete ML pipeline for trading:

- **fks_training**: Model training and management with MLflow
- **fks_api**: ML inference endpoints
- **fks_app**: Backtesting and strategy integration
- **fks_ai**: Sentiment analysis and advanced AI features

## Training Models

### Using MLflow Projects

The recommended way to train models is using MLflow Projects:

```bash
# Train LSTM model
mlflow run . -e train \
  --experiment-name lstm_forecasting \
  -P model_name=lstm_price_forecast \
  -P symbol=BTC/USDT \
  -P sequence_length=60 \
  -P epochs=100 \
  -P learning_rate=0.001

# Train with custom data
mlflow run . -e train \
  -P data_path=/path/to/data.csv \
  -P interval=1h \
  -P days_back=90
```

### Programmatic Training

```python
from src.domain.ml.training.train import load_data, main
from src.domain.ml.models.lstm.model import LSTMModelConfig, LSTMTrainer
from torch.utils.data import DataLoader

# Load data
df = load_data("BTC/USDT", interval="1h", days_back=30)

# Create model config
config = LSTMModelConfig(
    sequence_length=60,
    input_features=5,
    hidden_units=50,
    epochs=100,
)

# Train model
trainer = LSTMTrainer(config)
history = trainer.train(train_loader, val_loader)
```

## Making Predictions

### Using the API

```python
import httpx

response = httpx.post(
    "http://fks-api:8000/api/v1/ml/predict",
    json={
        "symbol": "BTC/USDT",
        "sequences": [[...]],  # Your input sequences
        "model_name": "lstm_price_forecast",
        "model_version": "latest",
        "include_sentiment": True,  # Optional sentiment integration
    }
)

result = response.json()
print(f"Prediction: {result['predictions'][0]}")
print(f"Confidence: {result['confidence_intervals'][0]}")
print(f"Feature Importance: {result['feature_importance']}")
```

### Direct Model Loading

```python
import mlflow

# Load model from MLflow
model_uri = "models:/lstm_price_forecast/latest"
model = mlflow.pyfunc.load_model(model_uri)

# Make prediction
prediction = model.predict(sequences)
```

## Model Management

### Listing Models

```python
import mlflow

client = mlflow.tracking.MlflowClient()
models = client.search_registered_models()

for model in models:
    print(f"Model: {model.name}")
    print(f"Latest Version: {model.latest_versions[0].version}")
```

### Promoting Models

```python
# Promote model to production
mlflow run . -e deploy \
  -P model_uri=runs:/<run_id>/model \
  -P model_name=lstm_price_forecast \
  -P stage=production \
  -P action=promote
```

## Backtesting with ML

### Simple Backtest

```python
from src.domain.trading.backtesting import BacktraderEngine, MLBacktraderStrategy

# Create engine
engine = BacktraderEngine(initial_cash=10000, commission=0.001)

# Run backtest
results = engine.run_backtest(
    data=historical_data,
    strategy=MLBacktraderStrategy,
    strategy_params={
        "model_name": "lstm_price_forecast",
        "model_version": "latest",
    }
)

print(f"Total Return: {results['total_return_pct']:.2f}%")
print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
```

### Walk-Forward Analysis

```python
from src.domain.trading.backtesting import WalkForwardAnalyzer

analyzer = WalkForwardAnalyzer(
    train_period_days=180,
    test_period_days=30,
    step_days=30,
)

results = analyzer.analyze(
    data=historical_data,
    strategy=MLBacktraderStrategy,
)

print(f"Average Test Return: {results['summary']['test']['avg_return']:.2%}")
print(f"Return Degradation: {results['summary']['degradation']['avg_return_degradation']:.2%}")
```

## Sentiment Integration

### Hybrid Signals

```python
from src.domain.trading.sentiment import SentimentMLStrategy

strategy = SentimentMLStrategy(
    sentiment_weight=0.3,
    ml_weight=0.7,
    min_confidence=0.6,
)

signal = await strategy.get_hybrid_signal(
    symbol="BTC/USDT",
    sequences=sequences,
    model_name="lstm_price_forecast",
)

print(f"Signal: {signal['signal']}")
print(f"Confidence: {signal['confidence']}")
print(f"ML Prediction: {signal['price_prediction']}")
print(f"Sentiment Score: {signal['sentiment_score']}")
```

## Advanced Models

### CNN-LSTM for Multi-Asset Analysis

```python
from src.domain.ml.models.cnn_lstm.model import (
    CNNLSTMModel,
    CNNLSTMConfig,
    CNNLSTMTrainer,
)

config = CNNLSTMConfig(
    sequence_length=60,
    num_assets=5,  # Analyze 5 correlated assets
    input_features=5,
    cnn_filters=64,
    lstm_hidden_units=50,
)

trainer = CNNLSTMTrainer(config)
history = trainer.train(train_loader, val_loader)
```

### Transformer for Long-Range Dependencies

```python
from src.domain.ml.models.transformer.model import (
    TransformerModel,
    TransformerConfig,
    TransformerTrainer,
)

config = TransformerConfig(
    sequence_length=120,  # Longer sequences
    input_features=5,
    d_model=128,
    nhead=8,
    num_layers=4,
)

trainer = TransformerTrainer(config)
history = trainer.train(train_loader, val_loader)
```

## Best Practices

1. **Data Quality**: Always validate input data before training
2. **Walk-Forward Validation**: Use walk-forward analysis to prevent overfitting
3. **Model Monitoring**: Regularly check for data drift and model degradation
4. **Confidence Thresholds**: Use confidence intervals to filter low-quality predictions
5. **Sentiment Weighting**: Adjust sentiment weights based on market conditions

## Troubleshooting

### Model Not Found

```python
# Check model registry
client = mlflow.tracking.MlflowClient()
models = client.search_registered_models(filter_string="name='lstm_price_forecast'")
```

### Low Prediction Confidence

- Check data quality and feature engineering
- Verify model was trained on similar data
- Consider retraining with more recent data

### Backtest Errors

- Ensure data has required columns: open, high, low, close, volume
- Check datetime index is properly formatted
- Verify sufficient historical data for sequence length

## Additional Resources

- [MLflow Documentation](https://www.mlflow.org/docs/latest/index.html)
- [Backtrader Documentation](https://www.backtrader.com/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

