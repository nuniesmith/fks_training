# FKS Training

Orchestrates model training pipelines and GPU resource allocation.

**Port**: 8011  
**Framework**: Python 3.12 + FastAPI  
**Role**: Model training pipelines and GPU resource allocation

## 🎯 Purpose

FKS Training orchestrates machine learning model training pipelines for the FKS Trading Platform. It provides:

- **Training Orchestration**: Manages training jobs and pipelines
- **GPU Resource Allocation**: Efficient GPU utilization for model training
- **Experiment Tracking**: MLflow integration for experiment tracking
- **Dataset Management**: Dataset versioning and management
- **Distributed Training**: Support for distributed training strategies

## 🏗️ Architecture

```
┌─────────────┐     ┌─────────────┐
│  fks_ai     │────▶│ fks_training│
│  (ML Models)│     │ (Orchestrator)│
└─────────────┘     └──────┬───────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   GPU Node  │
                    │  (Training) │
                    └──────┬───────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  MLflow     │
                    │  (Tracking) │
                    └─────────────┘
```

## 🚀 Quick Start

### Development

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install .[ml]

# Run service
python -m fks_training.main

# Or using uvicorn
uvicorn src.main:app --reload --host 0.0.0.0 --port 8011
```

### Docker

```bash
# Build and run
docker-compose up --build

# Or using the unified start script
cd /home/jordan/Documents/code/fks
./start.sh --type compose
```

### Kubernetes

```bash
# Deploy to Kubernetes
cd /home/jordan/Documents/code/fks
./start.sh --type k8s
```

## 📡 API Endpoints

### Health Checks

- `GET /health` - Health check
- `GET /ready` - Readiness check (checks GPU availability)
- `GET /live` - Liveness probe

### Training Jobs

- `POST /api/v1/training/jobs` - Create training job
- `GET /api/v1/training/jobs` - List all training jobs
- `GET /api/v1/training/jobs/{job_id}` - Get job status
- `DELETE /api/v1/training/jobs/{job_id}` - Cancel job
- `GET /api/v1/training/jobs/{job_id}/logs` - Get job logs

### Experiments

- `GET /api/v1/experiments` - List experiments
- `GET /api/v1/experiments/{experiment_id}` - Get experiment details
- `GET /api/v1/experiments/{experiment_id}/runs` - Get experiment runs

### GPU Resources

- `GET /api/v1/gpu/status` - Get GPU status
- `GET /api/v1/gpu/allocations` - Get current GPU allocations

## 🔧 Configuration

### Environment Variables

```bash
# Service Configuration
SERVICE_NAME=fks_training
SERVICE_PORT=8011

# MLflow Configuration
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_EXPERIMENT_NAME=fks_trading

# GPU Configuration
CUDA_VISIBLE_DEVICES=0,1  # Which GPUs to use
GPU_MEMORY_FRACTION=0.8    # Fraction of GPU memory to use

# Training Configuration
TRAINING_DATA_PATH=/data/training
MODEL_OUTPUT_PATH=/data/models
MAX_CONCURRENT_JOBS=2

# Dataset Configuration
DATASET_CACHE_PATH=/data/cache
DATASET_VERSIONING=true

# Distributed Training
DISTRIBUTED_STRATEGY=mirrored  # mirrored, parameter_server, multi_worker
NUM_WORKERS=1
```

## 🧪 Testing

```bash
# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Integration tests (requires GPU)
pytest tests/integration/ -v
```

## 🐳 Docker

### Build

```bash
docker build -t nuniesmith/fks:training-latest .
```

### Run

```bash
docker run -p 8011:8011 \
  --gpus all \
  -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
  -v /data/training:/data/training \
  nuniesmith/fks:training-latest
```

### Docker Compose

```yaml
services:
  fks_training:
    build: .
    image: nuniesmith/fks:training-latest
    ports:
      - "8011:8011"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    volumes:
      - training_data:/data/training
```

## ☸️ Kubernetes

### Deployment

```bash
# Deploy using Helm
cd repo/main/k8s/charts/fks-platform
helm install fks-platform . -n fks-trading

# Or using the unified start script
cd /home/jordan/Documents/code/fks
./start.sh --type k8s
```

### GPU Node Selection

The service should be deployed on GPU-enabled nodes:

```yaml
nodeSelector:
  accelerator: nvidia-tesla-k80
tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
```

### Health Checks

Kubernetes probes:
- **Liveness**: `GET /live`
- **Readiness**: `GET /ready` (checks GPU availability)

## 📚 Documentation

- [API Documentation](docs/API.md) - Complete API reference
- [Training Guide](docs/TRAINING_GUIDE.md) - Training pipeline documentation
- [GPU Configuration](docs/GPU_CONFIG.md) - GPU setup and optimization

## 🔗 Integration

### Dependencies

- **fks_ai**: ML models and architectures
- **fks_data**: Training datasets
- **MLflow**: Experiment tracking
- **GPU Nodes**: For model training

### Consumers

- **fks_ai**: Consumes trained models
- **fks_web**: Training job management interface

## 📊 Monitoring

### Health Check Endpoints

- `GET /health` - Basic health status
- `GET /ready` - Readiness (checks GPU availability)
- `GET /live` - Liveness (process alive)

### Metrics

- Training job completion rates
- GPU utilization
- Training duration and throughput
- Model performance metrics
- Dataset processing rates

### Logging

- Training job logs
- GPU allocation tracking
- Experiment tracking events
- Error tracking and retries

## 🛠️ Development

### Setup

```bash
# Clone repository
git clone https://github.com/nuniesmith/fks_training.git
cd fks_training

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install .[ml,dev]
```

### Code Structure

```
repo/training/
├── src/
│   ├── main.py              # FastAPI application
│   ├── services/
│   │   ├── training.py      # Training orchestration
│   │   ├── gpu_manager.py  # GPU resource management
│   │   └── mlflow_client.py # MLflow integration
│   ├── pipelines/           # Training pipelines
│   │   ├── base.py         # Base pipeline
│   │   └── trading.py      # Trading model pipeline
│   └── api/
│       └── routes/          # API routes
├── tests/                   # Test suite
├── Dockerfile              # Container definition
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

### Contributing

1. Follow Python best practices (PEP 8)
2. Write tests for training pipelines
3. Document GPU requirements
4. Update MLflow experiment configurations

## 🔄 Training Pipeline

1. **Dataset Preparation**: Load and preprocess data
2. **Model Initialization**: Load model architecture
3. **Training**: Execute training loop
4. **Validation**: Evaluate on validation set
5. **Checkpointing**: Save model checkpoints
6. **Logging**: Track metrics in MLflow
7. **Model Export**: Export trained model

## 🐛 Troubleshooting

### GPU Not Available

```bash
# Check GPU availability
nvidia-smi

# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
```

### MLflow Connection Issues

- Verify MLflow server is running
- Check `MLFLOW_TRACKING_URI` environment variable
- Ensure network connectivity to MLflow server

### Out of Memory

- Reduce batch size
- Use gradient accumulation
- Enable mixed precision training
- Reduce model size

---

**Repository**: [nuniesmith/fks_training](https://github.com/nuniesmith/fks_training)  
**Docker Image**: `nuniesmith/fks:training-latest`  
**Status**: Active Development
