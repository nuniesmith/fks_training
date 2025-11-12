"""Local shim for shared_python dependency (logging, exceptions minimal)."""

from .logging import get_logger  # type: ignore
from .exceptions import ModelError  # type: ignore

__all__ = ["get_logger", "ModelError"]
