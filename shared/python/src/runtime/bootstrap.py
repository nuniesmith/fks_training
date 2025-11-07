from __future__ import annotations
from .registry import get_app
from .registry import list_apps as _list_apps
from .registry import register_app as _register_app
from .registry import AppSpec  # noqa: F401
from . import registry  # noqa
from . import __init__ as _runtime_init  # noqa
from shared_python import init_logging, get_logger, get_settings, initialize_fks_service  # type: ignore

def run_app(name: str) -> None:
    """Initialize logging/settings and invoke the registered app entrypoint."""
    init_logging()
    settings = initialize_fks_service()
    logger = get_logger(f"shared_python.{name}")
    logger.info("Launching app", extra={"app": name, "env": settings.APP_ENV})
    spec = get_app(name)
    spec.entrypoint()

__all__ = ["run_app"]
