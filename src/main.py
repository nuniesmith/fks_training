"""
TRAINING Service Entry Point

This module serves as the entry point for the TRAINING service.
Provides GPU-enabled training infrastructure for ML models.
"""

import os
import sys
import logging

from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FKS Training Service",
    description="GPU-enabled training infrastructure for ML models",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up Prometheus metrics with fks_build_info
try:
    from prometheus_client import CollectorRegistry, Gauge, generate_latest
    
    _metrics_registry = CollectorRegistry()
    _build_info = Gauge(
        "fks_build_info",
        "Build information for the service",
        ["service", "version"],
        registry=_metrics_registry,
    )
    _build_info.labels(service="fks_training", version="1.0.0").set(1)
    
    @app.get("/metrics", response_class=PlainTextResponse, include_in_schema=False)
    async def metrics_endpoint():
        return PlainTextResponse(
            generate_latest(_metrics_registry).decode("utf-8"),
            media_type="text/plain; version=0.0.4; charset=utf-8"
        )
    logger.info("✅ Prometheus metrics with fks_build_info registered")
except Exception as e:
    logger.warning(f"Could not set up Prometheus metrics: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse({
        "status": "healthy",
        "service": "fks_training",
        "version": "1.0.0"
    })


@app.get("/")
async def root():
    """Root endpoint"""
    return JSONResponse({
        "service": "fks_training",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics"
        }
    })


def main():
    import uvicorn
    service_name = os.getenv("SERVICE_NAME", "fks_training")
    port = int(os.getenv("SERVICE_PORT", "8011"))
    logger.info(f"Starting {service_name} service on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    sys.exit(main())
