"""
Model Performance Monitoring

Tracks model performance over time and alerts on degradation.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import mlflow
import numpy as np

logger = logging.getLogger(__name__)


class ModelPerformanceMonitor:
    """
    Monitor model performance in production.

    Tracks metrics, detects drift, and generates alerts.
    """

    def __init__(
        self,
        model_name: str,
        baseline_metrics: Dict[str, float],
        tracking_uri: Optional[str] = None,
    ):
        """
        Initialize performance monitor.

        Args:
            model_name: Name of the model being monitored
            baseline_metrics: Baseline performance metrics
            tracking_uri: MLflow tracking URI (optional)
        """
        self.model_name = model_name
        self.baseline_metrics = baseline_metrics
        self.tracking_uri = tracking_uri

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # Performance history
        self.metric_history: Dict[str, List[float]] = {}
        self.timestamps: List[datetime] = []

    def log_prediction(
        self,
        prediction: float,
        actual: Optional[float] = None,
        features: Optional[np.ndarray] = None,
    ) -> None:
        """
        Log a prediction for monitoring.

        Args:
            prediction: Model prediction
            actual: Actual value (if available)
            features: Input features (for drift detection)
        """
        timestamp = datetime.utcnow()
        self.timestamps.append(timestamp)

        # Calculate error if actual available
        if actual is not None:
            error = abs(prediction - actual)
            self._update_metric("prediction_error", error)

            # Calculate accuracy metrics
            if len(self.metric_history.get("prediction_error", [])) > 0:
                mae = np.mean(self.metric_history["prediction_error"][-100:])  # Last 100
                self._update_metric("mae", mae)

    def _update_metric(self, metric_name: str, value: float) -> None:
        """Update metric history."""
        if metric_name not in self.metric_history:
            self.metric_history[metric_name] = []
        self.metric_history[metric_name].append(value)

        # Keep only last 1000 values
        if len(self.metric_history[metric_name]) > 1000:
            self.metric_history[metric_name] = self.metric_history[metric_name][-1000:]

    def get_current_metrics(self, window: int = 100) -> Dict[str, float]:
        """
        Get current performance metrics over a window.

        Args:
            window: Number of recent predictions to include

        Returns:
            Dictionary of current metrics
        """
        metrics = {}

        for metric_name, values in self.metric_history.items():
            if len(values) > 0:
                recent_values = values[-window:]
                metrics[f"{metric_name}_mean"] = float(np.mean(recent_values))
                metrics[f"{metric_name}_std"] = float(np.std(recent_values))
                metrics[f"{metric_name}_p95"] = float(np.percentile(recent_values, 95))
                metrics[f"{metric_name}_p99"] = float(np.percentile(recent_values, 99))

        return metrics

    def check_performance_degradation(self) -> Dict[str, any]:
        """
        Check if model performance has degraded.

        Returns:
            Dictionary with degradation analysis
        """
        current_metrics = self.get_current_metrics()

        # Compare to baseline
        degradation = {}
        for metric_name, baseline_value in self.baseline_metrics.items():
            current_key = f"{metric_name}_mean"
            if current_key in current_metrics:
                current_value = current_metrics[current_key]

                if baseline_value != 0:
                    degradation_ratio = abs(current_value - baseline_value) / abs(baseline_value)
                else:
                    degradation_ratio = abs(current_value - baseline_value)

                degradation[metric_name] = {
                    "baseline": float(baseline_value),
                    "current": float(current_value),
                    "degradation_ratio": float(degradation_ratio),
                }

        # Determine if significant degradation
        significant_degradation = any(
            d["degradation_ratio"] > 0.2 for d in degradation.values()
        )

        return {
            "degradation_detected": significant_degradation,
            "degradation_details": degradation,
            "current_metrics": current_metrics,
        }

    def log_to_mlflow(self, run_name: Optional[str] = None) -> None:
        """
        Log current metrics to MLflow.

        Args:
            run_name: Optional run name
        """
        if not self.tracking_uri:
            logger.warning("No tracking URI configured, skipping MLflow logging")
            return

        try:
            run_name = run_name or f"{self.model_name}_monitoring_{datetime.utcnow().isoformat()}"
            with mlflow.start_run(run_name=run_name):
                current_metrics = self.get_current_metrics()
                mlflow.log_metrics(current_metrics)
                mlflow.log_param("model_name", self.model_name)
                mlflow.log_param("monitoring_timestamp", datetime.utcnow().isoformat())

                # Log degradation check
                degradation_check = self.check_performance_degradation()
                mlflow.log_param("degradation_detected", degradation_check["degradation_detected"])

        except Exception as e:
            logger.error(f"Failed to log to MLflow: {e}")

