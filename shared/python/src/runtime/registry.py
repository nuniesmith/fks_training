from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List

@dataclass(slots=True)
class AppSpec:
    name: str
    entrypoint: Callable[[], None]
    description: str = ""

_REGISTRY: Dict[str, AppSpec] = {}

def register_app(name: str, entrypoint: Callable[[], None], description: str = "") -> AppSpec:
    if name in _REGISTRY:  # pragma: no cover
        raise ValueError(f"App '{name}' already registered")
    spec = AppSpec(name=name, entrypoint=entrypoint, description=description)
    _REGISTRY[name] = spec
    return spec

def list_apps() -> List[str]:
    return sorted(_REGISTRY.keys())

def get_app(name: str) -> AppSpec:
    try:
        return _REGISTRY[name]
    except KeyError:  # pragma: no cover
        raise KeyError(f"Unknown app '{name}'. Available: {list_apps()}")

__all__ = ["register_app", "list_apps", "get_app", "AppSpec"]
