"""Worker app registration using existing template service.

Bridges the legacy `start_template_service` into the new runtime registry API.
"""
from __future__ import annotations
from runtime.registry import register_app

def _run_worker():  # pragma: no cover (thin wrapper)
    try:
        from framework.services.template import start_template_service  # type: ignore
    except Exception:  # pragma: no cover
        print("[shared_python.worker] Missing framework.services.template, cannot start worker")
        return

    import os

    service_name = os.getenv("WORKER_SERVICE_NAME", "worker")
    port = int(os.getenv("WORKER_SERVICE_PORT", os.getenv("SERVICE_PORT", "8006")))
    start_template_service(service_name=service_name, service_port=port)

register_app("worker", _run_worker, description="FKS Worker service")

__all__ = []
