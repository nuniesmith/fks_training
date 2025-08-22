"""
TRAINING Service Entry Point

This module serves as the entry point for the TRAINING service, integrating with the main application
and utilizing the service template for service management.
"""

import os
import sys

from framework.services.template import start_template_service


def main():
    # Set the service name and port from environment variables or defaults
    service_name = os.getenv("TRAINING_SERVICE_NAME", "training")
    port = os.getenv("TRAINING_SERVICE_PORT", "8088")

    # Log the service startup
    print(f"Starting {service_name} service on port {port}")

    # Start the service using the template
    start_template_service(service_name=service_name, service_port=port)


if __name__ == "__main__":
    sys.exit(main())
