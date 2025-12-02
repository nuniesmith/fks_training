"""
TRAINING Service Entry Point

This module serves as the entry point for the TRAINING service, integrating with the main application
and utilizing the service template for service management.
"""

import os
import sys

try:
    from framework.services.template import start_template_service  # type: ignore
    _HAS_TEMPLATE = True
except Exception:  # pragma: no cover
    _HAS_TEMPLATE = False
    def start_template_service(*args, **kwargs):  # type: ignore
        print("[fks_training.main] framework.services.template missing - fallback noop")


def main():
    # Use SERVICE_PORT from environment (defaults to 8011)
    service_name = os.getenv("SERVICE_NAME", "fks_training")
    port = int(os.getenv("SERVICE_PORT", "8011"))
    print(f"[fks_training] Starting {service_name} service on port {port}")
    
    if not _HAS_TEMPLATE:
        raise RuntimeError(
            "FastAPI template service not available. "
            "Please ensure framework.services.template is installed."
        )
    
    try:
        start_template_service(service_name=service_name, service_port=port)
    except Exception as e:
        print(f"[fks_training] Failed to start FastAPI service: {e}")
        raise


if __name__ == "__main__":
    sys.exit(main())
