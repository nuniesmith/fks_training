"""
Model Explainability Tools

Provides SHAP and LIME explanations for model predictions.
"""

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Try to import SHAP
try:
    import shap
    _shap_available = True
except ImportError:
    _shap_available = False
    logger.warning("SHAP not available")

# Try to import LIME
try:
    import lime
    import lime.lime_tabular
    _lime_available = True
except ImportError:
    _lime_available = False
    logger.warning("LIME not available")


class ModelExplainer:
    """
    Provide explanations for ML model predictions.

    Supports:
    - SHAP (SHapley Additive exPlanations)
    - LIME (Local Interpretable Model-agnostic Explanations)
    """

    def __init__(self, model, background_data: Optional[np.ndarray] = None):
        """
        Initialize explainer.

        Args:
            model: Trained ML model with predict() method
            background_data: Background dataset for SHAP (optional)
        """
        self.model = model
        self.background_data = background_data
        self.shap_available = _shap_available
        self.lime_available = _lime_available

        # Initialize SHAP explainer if available
        self.shap_explainer = None
        if self.shap_available and background_data is not None:
            try:
                # Try TreeExplainer for tree models
                self.shap_explainer = shap.TreeExplainer(model)
            except Exception:
                try:
                    # Fallback to KernelExplainer
                    self.shap_explainer = shap.KernelExplainer(
                        model.predict, background_data[:100]  # Use subset for speed
                    )
                except Exception:
                    logger.warning("Could not initialize SHAP explainer")

    def explain_shap(
        self, instances: np.ndarray, feature_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Generate SHAP explanations.

        Args:
            instances: Input instances to explain
            feature_names: Optional feature names

        Returns:
            Dictionary with SHAP values and feature importance
        """
        if not self.shap_available or self.shap_explainer is None:
            raise ValueError("SHAP not available or explainer not initialized")

        # Calculate SHAP values
        shap_values = self.shap_explainer.shap_values(instances)

        # Handle multi-output models
        if isinstance(shap_values, list):
            # Average across outputs
            shap_values = np.mean(shap_values, axis=0)

        # Calculate feature importance (mean absolute SHAP values)
        feature_importance = np.abs(shap_values).mean(axis=0)

        # Create feature importance dictionary
        if feature_names:
            importance_dict = {
                name: float(importance)
                for name, importance in zip(feature_names, feature_importance)
            }
        else:
            importance_dict = {
                f"feature_{i}": float(importance)
                for i, importance in enumerate(feature_importance)
            }

        return {
            "shap_values": shap_values.tolist() if isinstance(shap_values, np.ndarray) else shap_values,
            "feature_importance": importance_dict,
            "explainer_type": "SHAP",
        }

    def explain_lime(
        self,
        instance: np.ndarray,
        feature_names: Optional[List[str]] = None,
        num_features: int = 10,
    ) -> Dict:
        """
        Generate LIME explanations.

        Args:
            instance: Single instance to explain
            feature_names: Optional feature names
            num_features: Number of top features to return

        Returns:
            Dictionary with LIME explanations
        """
        if not self.lime_available:
            raise ValueError("LIME not available")

        if self.background_data is None:
            raise ValueError("Background data required for LIME")

        # Create LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            self.background_data,
            feature_names=feature_names,
            mode="regression",  # or "classification" based on model type
        )

        # Generate explanation
        explanation = explainer.explain_instance(
            instance.flatten() if len(instance.shape) > 1 else instance,
            self.model.predict,
            num_features=num_features,
        )

        # Extract feature contributions
        feature_contributions = {
            feature: contribution
            for feature, contribution in explanation.as_list()
        }

        return {
            "feature_contributions": feature_contributions,
            "explainer_type": "LIME",
        }

    def explain(
        self,
        instances: np.ndarray,
        method: str = "shap",
        feature_names: Optional[List[str]] = None,
    ) -> Dict:
        """
        Generate explanations using specified method.

        Args:
            instances: Input instances to explain
            method: Explanation method ("shap" or "lime")
            feature_names: Optional feature names

        Returns:
            Dictionary with explanations
        """
        if method == "shap":
            return self.explain_shap(instances, feature_names)
        elif method == "lime":
            if len(instances.shape) > 1 and instances.shape[0] > 1:
                # LIME works on single instances, take first
                return self.explain_lime(instances[0], feature_names)
            return self.explain_lime(instances, feature_names)
        else:
            raise ValueError(f"Unknown explanation method: {method}")

