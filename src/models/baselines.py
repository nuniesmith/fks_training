"""Baseline models: wrappers for XGBoost / CatBoost with uniform interface.

These lightweight wrappers defer heavy imports until fit/predict to keep startup light.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
from shared_python.logging import get_logger  # type: ignore
from shared_python.exceptions import ModelError  # type: ignore

log = get_logger(__name__)


@dataclass
class TrainResult:
    model: Any
    params: dict
    training_time_s: float | None = None


class XGBoostClassifierWrapper:
    def __init__(self, **params: Any):
        self.params = params
        self._model: Any | None = None

    def fit(self, X, y) -> TrainResult:  # type: ignore
        import time
        t0 = time.time()
        try:
            import xgboost as xgb  # type: ignore
            self._model = xgb.XGBClassifier(**self.params)
            self._model.fit(X, y)
        except Exception as e:  # pragma: no cover
            raise ModelError(f"XGBoost fit failed: {e}") from e
        return TrainResult(model=self._model, params=self.params, training_time_s=time.time() - t0)

    def predict_proba(self, X):  # type: ignore
        if self._model is None:
            raise ModelError("Model not fitted")
        return self._model.predict_proba(X)


class CatBoostClassifierWrapper:
    def __init__(self, **params: Any):
        self.params = params
        self._model: Any | None = None

    def fit(self, X, y) -> TrainResult:  # type: ignore
        import time
        t0 = time.time()
        try:
            from catboost import CatBoostClassifier  # type: ignore
            self._model = CatBoostClassifier(verbose=False, **self.params)
            self._model.fit(X, y)
        except Exception as e:  # pragma: no cover
            raise ModelError(f"CatBoost fit failed: {e}") from e
        return TrainResult(model=self._model, params=self.params, training_time_s=time.time() - t0)

    def predict_proba(self, X):  # type: ignore
        if self._model is None:
            raise ModelError("Model not fitted")
        return self._model.predict_proba(X)
