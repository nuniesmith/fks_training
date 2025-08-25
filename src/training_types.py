"""Common type aliases for training pipelines.

Renamed from types.py to avoid shadowing the Python stdlib module 'types'.
"""
from typing import Any, Mapping, Protocol

Params = Mapping[str, Any]

class Trainable(Protocol):
    def fit(self, X: Any, y: Any | None = None, **kwargs: Any) -> "Trainable": ...

__all__ = ["Params", "Trainable"]
