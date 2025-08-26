# Dockerfile Options

This service supports two build strategies:

1. Simplified: 
   docker build -f Dockerfile.simple -t fks_training:simple .

2. Unified (full multi-runtime + GPU support) shared file:
   docker build -f ../../shared/shared_docker/Dockerfile -t fks_training:unified      --build-arg SERVICE_RUNTIME=python .

Default repository Dockerfile may still be the unified version; adopt the simple one by renaming if desired.
