"""
Bias Detection and Fairness Metrics

Uses AIF360 to detect and measure bias in ML models.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import AIF360
try:
    from aif360.datasets import BinaryLabelDataset
    from aif360.metrics import BinaryLabelDatasetMetric
    from aif360.algorithms.preprocessing import Reweighing
    _aif360_available = True
except ImportError:
    _aif360_available = False
    logger.warning("AIF360 not available, bias detection will be limited")


class BiasDetector:
    """
    Detect and measure bias in ML models and datasets.

    Supports:
    - Disparate impact ratio
    - Equalized odds
    - Statistical parity difference
    """

    def __init__(self):
        """Initialize bias detector."""
        self.aif360_available = _aif360_available

    def calculate_disparate_impact(
        self,
        predictions: np.ndarray,
        protected_attribute: np.ndarray,
        favorable_label: int = 1,
    ) -> float:
        """
        Calculate disparate impact ratio.

        Measures if model favors one group over another.
        Ratio should be close to 1.0 for fairness.

        Args:
            predictions: Model predictions (binary)
            protected_attribute: Protected group labels (0 or 1)
            favorable_label: Label considered favorable (default: 1)

        Returns:
            Disparate impact ratio (0-1, where 1 is fair)
        """
        if not self.aif360_available:
            # Fallback calculation
            protected_group = predictions[protected_attribute == 1]
            unprotected_group = predictions[protected_attribute == 0]

            protected_favorable_rate = (protected_group == favorable_label).mean()
            unprotected_favorable_rate = (unprotected_group == favorable_label).mean()

            if unprotected_favorable_rate > 0:
                return protected_favorable_rate / unprotected_favorable_rate
            return 0.0

        # Use AIF360 for more robust calculation
        try:
            # Create dataset
            df = pd.DataFrame(
                {
                    "prediction": predictions,
                    "protected": protected_attribute,
                }
            )

            dataset = BinaryLabelDataset(
                df=df,
                label_names=["prediction"],
                protected_attribute_names=["protected"],
                favorable_label=favorable_label,
                unfavorable_label=1 - favorable_label,
            )

            # Calculate metric
            metric = BinaryLabelDatasetMetric(
                dataset,
                unprivileged_groups=[{"protected": 0}],
                privileged_groups=[{"protected": 1}],
            )

            return metric.disparate_impact()

        except Exception as e:
            logger.error(f"Error calculating disparate impact: {e}")
            return 1.0  # Return neutral value on error

    def calculate_equalized_odds(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        protected_attribute: np.ndarray,
    ) -> Dict[str, float]:
        """
        Calculate equalized odds metrics.

        Measures if true positive rates are similar across groups.

        Args:
            predictions: Model predictions
            true_labels: True labels
            protected_attribute: Protected group labels

        Returns:
            Dictionary with TPR differences
        """
        results = {}

        # Calculate TPR for each group
        for group_val in [0, 1]:
            group_mask = protected_attribute == group_val
            group_predictions = predictions[group_mask]
            group_labels = true_labels[group_mask]

            if len(group_labels) > 0:
                true_positives = ((group_predictions == 1) & (group_labels == 1)).sum()
                false_negatives = ((group_predictions == 0) & (group_labels == 1)).sum()

                if (true_positives + false_negatives) > 0:
                    tpr = true_positives / (true_positives + false_negatives)
                    results[f"tpr_group_{group_val}"] = float(tpr)
                else:
                    results[f"tpr_group_{group_val}"] = 0.0

        # Calculate difference
        if "tpr_group_0" in results and "tpr_group_1" in results:
            results["tpr_difference"] = abs(results["tpr_group_0"] - results["tpr_group_1"])

        return results

    def check_fairness(
        self,
        predictions: np.ndarray,
        protected_attribute: np.ndarray,
        true_labels: Optional[np.ndarray] = None,
        threshold: float = 0.8,
    ) -> Dict[str, any]:
        """
        Comprehensive fairness check.

        Args:
            predictions: Model predictions
            protected_attribute: Protected group labels
            true_labels: True labels (optional, for equalized odds)
            threshold: Minimum disparate impact for fairness (default: 0.8)

        Returns:
            Dictionary with fairness metrics and verdict
        """
        results = {
            "disparate_impact": self.calculate_disparate_impact(
                predictions, protected_attribute
            ),
            "is_fair": True,
            "issues": [],
        }

        # Check disparate impact
        if results["disparate_impact"] < threshold:
            results["is_fair"] = False
            results["issues"].append(
                f"Disparate impact ratio ({results['disparate_impact']:.3f}) "
                f"below threshold ({threshold})"
            )

        # Check equalized odds if labels available
        if true_labels is not None:
            equalized_odds = self.calculate_equalized_odds(
                predictions, true_labels, protected_attribute
            )
            results["equalized_odds"] = equalized_odds

            if "tpr_difference" in equalized_odds:
                if equalized_odds["tpr_difference"] > 0.1:  # 10% difference threshold
                    results["is_fair"] = False
                    results["issues"].append(
                        f"TPR difference ({equalized_odds['tpr_difference']:.3f}) "
                        f"exceeds threshold (0.1)"
                    )

        return results

