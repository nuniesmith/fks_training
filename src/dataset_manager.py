"""Dataset management utilities (placeholder).

Implement dataset loading, caching, and splitting here.
"""

from typing import Iterable, Protocol, Any


class Dataset(Protocol):
	def __iter__(self) -> Iterable[Any]: ...


def get_default_dataset() -> Dataset | None:
	"""Return a default dataset instance if configured, else None."""
	return None


__all__ = ["Dataset", "get_default_dataset"]

