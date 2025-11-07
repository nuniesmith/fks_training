"""Base training pipeline abstractions (placeholder)."""

from typing import Protocol, Any


class Pipeline(Protocol):
	def run(self, **kwargs: Any) -> Any: ...


__all__ = ["Pipeline"]

