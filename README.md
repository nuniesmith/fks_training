# FKS Training Service

Orchestrates model training pipelines and GPU resource allocation.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install .[ml]
python -m fks_training.main
```

## Next Steps

- Add experiment tracking (MLflow)
- Add dataset versioning
- Implement distributed training strategy
