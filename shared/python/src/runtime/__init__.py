"""Runtime registry & bootstrap for shared_python (worker phased migration)."""
from .registry import register_app, list_apps, get_app  # noqa: F401
from .bootstrap import run_app  # noqa: F401
__all__ = ["register_app", "list_apps", "get_app", "run_app"]
