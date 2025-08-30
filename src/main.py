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


def _fallback_service(service_name: str, port: int):  # minimal Flask for health & gpu info
    from flask import Flask, jsonify  # type: ignore
    app = Flask(service_name)

    @app.get("/health")
    def health():  # pragma: no cover - runtime path
        from gpu_manager import is_gpu_available
        return jsonify({
            "service": service_name,
            "status": "healthy",
            "gpu": is_gpu_available(),
            "port": port,
        })

    @app.get("/")
    def root():  # pragma: no cover
        return jsonify({"service": service_name, "status": "ok"})

    app.run(host="0.0.0.0", port=port, debug=False)


def main():
    # Standardize port to 8005 unless explicitly overridden
    service_name = os.getenv("TRAINING_SERVICE_NAME", "training")
    port = int(os.getenv("TRAINING_SERVICE_PORT", os.getenv("SERVICE_PORT", "8005")))
    print(f"[fks_training] Starting {service_name} service on port {port}")
    if _HAS_TEMPLATE:
        try:
            start_template_service(service_name=service_name, service_port=port)
            return
        except Exception as e:  # pragma: no cover
            print(f"[fks_training] template start failed: {e}; using fallback")
    _fallback_service(service_name, port)


if __name__ == "__main__":
    sys.exit(main())
