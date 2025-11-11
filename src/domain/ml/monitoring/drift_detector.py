"""
Data and Model Drift Detection

Detects when input data distribution changes or model performance degrades.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Try to import Alibi Detect for advanced drift detection
try:
    from alibi_detect import KSDrift, MMDDrift
    _alibi_available = True
except ImportError:
    _alibi_available = False
    logger.warning("Alibi Detect not available, using basic statistical tests")


class DataDriftDetector:
    """
    Detect data drift in input features.

    Compares current data distribution to reference (training) distribution.
    """

    def __init__(self, reference_data: np.ndarray, threshold: float = 0.05):
        """
        Initialize drift detector.

        Args:
            reference_data: Reference data (training data) of shape (n_samples, n_features)
            threshold: P-value threshold for drift detection (default: 0.05)
        """
        self.reference_data = reference_data
        self.threshold = threshold
        self.alibi_available = _alibi_available

        # Initialize Alibi detectors if available
        if self.alibi_available:
            try:
                self.ks_detector = KSDrift(reference_data, p_val=self.threshold)
                self.mmd_detector = MMDDrift(reference_data, p_val=self.threshold)
            except Exception as e:
                logger.warning(f"Could not initialize Alibi detectors: {e}")
                self.alibi_available = False

    def detect_drift(
        self, current_data: np.ndarray, method: str = "ks"
    ) -> Dict[str, any]:
        """
        Detect drift in current data compared to reference.

        Args:
            current_data: Current data of shape (n_samples, n_features)
            method: Detection method ('ks', 'mmd', or 'statistical')

        Returns:
            Dictionary with drift detection results
        """
        if method == "ks" and self.alibi_available:
            return self._detect_with_alibi_ks(current_data)
        elif method == "mmd" and self.alibi_available:
            return self._detect_with_alibi_mmd(current_data)
        else:
            return self._detect_with_statistical_tests(current_data)

    def _detect_with_alibi_ks(self, current_data: np.ndarray) -> Dict:
        """Detect drift using Alibi's KS test."""
        try:
            drift_pred = self.ks_detector.predict(current_data)
            return {
                "drift_detected": bool(drift_pred["data"]["is_drift"]),
                "p_value": float(drift_pred["data"]["p_val"]),
                "threshold": self.threshold,
                "method": "ks",
                "distance": float(drift_pred["data"]["distance"]) if "distance" in drift_pred["data"] else None,
            }
        except Exception as e:
            logger.error(f"Alibi KS drift detection failed: {e}")
            return self._detect_with_statistical_tests(current_data)

    def _detect_with_alibi_mmd(self, current_data: np.ndarray) -> Dict:
        """Detect drift using Alibi's MMD test."""
        try:
            drift_pred = self.mmd_detector.predict(current_data)
            return {
                "drift_detected": bool(drift_pred["data"]["is_drift"]),
                "p_value": float(drift_pred["data"]["p_val"]),
                "threshold": self.threshold,
                "method": "mmd",
                "distance": float(drift_pred["data"]["distance"]) if "distance" in drift_pred["data"] else None,
            }
        except Exception as e:
            logger.error(f"Alibi MMD drift detection failed: {e}")
            return self._detect_with_statistical_tests(current_data)

    def _detect_with_statistical_tests(self, current_data: np.ndarray) -> Dict:
        """Detect drift using basic statistical tests."""
        results = {
            "drift_detected": False,
            "method": "statistical",
            "threshold": self.threshold,
            "feature_drift": {},
        }

        n_features = self.reference_data.shape[1]
        drift_count = 0

        for i in range(n_features):
            ref_feature = self.reference_data[:, i]
            curr_feature = current_data[:, i]

            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = stats.ks_2samp(ref_feature, curr_feature)

            # Mann-Whitney U test (non-parametric)
            mw_stat, mw_pvalue = stats.mannwhitneyu(ref_feature, curr_feature, alternative="two-sided")

            # Calculate mean difference
            mean_diff = abs(np.mean(curr_feature) - np.mean(ref_feature))
            std_diff = abs(np.std(curr_feature) - np.std(ref_feature))

            feature_drift = {
                "ks_pvalue": float(ks_pvalue),
                "mw_pvalue": float(mw_pvalue),
                "mean_difference": float(mean_diff),
                "std_difference": float(std_diff),
                "drift_detected": ks_pvalue < self.threshold or mw_pvalue < self.threshold,
            }

            results["feature_drift"][f"feature_{i}"] = feature_drift

            if feature_drift["drift_detected"]:
                drift_count += 1

        # Overall drift if significant number of features show drift
        results["drift_detected"] = drift_count > (n_features * 0.3)  # 30% of features
        results["drift_ratio"] = drift_count / n_features

        return results


class ModelDriftDetector:
    """
    Detect model performance drift.

    Monitors prediction accuracy and detects when model performance degrades.
    """

    def __init__(
        self,
        baseline_metrics: Dict[str, float],
        threshold_ratio: float = 0.2,  # 20% degradation threshold
    ):
        """
        Initialize model drift detector.

        Args:
            baseline_metrics: Baseline performance metrics (from training/validation)
            threshold_ratio: Ratio of degradation to trigger drift (default: 0.2)
        """
        self.baseline_metrics = baseline_metrics
        self.threshold_ratio = threshold_ratio

    def detect_drift(self, current_metrics: Dict[str, float]) -> Dict[str, any]:
        """
        Detect model performance drift.

        Args:
            current_metrics: Current performance metrics

        Returns:
            Dictionary with drift detection results
        """
        results = {
            "drift_detected": False,
            "metric_drift": {},
            "overall_drift": False,
        }

        drift_count = 0
        total_metrics = 0

        for metric_name, baseline_value in self.baseline_metrics.items():
            if metric_name not in current_metrics:
                continue

            current_value = current_metrics[metric_name]
            total_metrics += 1

            # Calculate degradation
            if baseline_value != 0:
                degradation = abs(current_value - baseline_value) / abs(baseline_value)
            else:
                degradation = abs(current_value - baseline_value)

            # Determine if metric should increase or decrease
            # For loss metrics (mse, mae, etc.), lower is better
            # For accuracy metrics (r2, accuracy, etc.), higher is better
            is_loss_metric = any(
                loss_term in metric_name.lower()
                for loss_term in ["loss", "mse", "mae", "rmse", "error"]
            )

            if is_loss_metric:
                # For loss metrics, drift if current > baseline * (1 + threshold)
                drift = current_value > baseline_value * (1 + self.threshold_ratio)
            else:
                # For accuracy metrics, drift if current < baseline * (1 - threshold)
                drift = current_value < baseline_value * (1 - self.threshold_ratio)

            metric_drift = {
                "baseline": float(baseline_value),
                "current": float(current_value),
                "degradation": float(degradation),
                "drift_detected": drift,
            }

            results["metric_drift"][metric_name] = metric_drift

            if drift:
                drift_count += 1

        # Overall drift if significant metrics show degradation
        if total_metrics > 0:
            drift_ratio = drift_count / total_metrics
            results["drift_ratio"] = drift_ratio
            results["overall_drift"] = drift_ratio > 0.3  # 30% of metrics degraded

        return results

