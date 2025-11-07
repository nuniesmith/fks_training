"""App adapters namespace.

Importing this module triggers registration of available apps. For now we
only register the worker service as a proof-of-concept.
"""
from .worker import *  # noqa: F401,F403

__all__ = []
