from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Dict, Optional

import mlflow
import torch
from framework.base import BaseService
from framework.patterns.space_based import Space
from models.registry import ModelRegistry
from services.training.dataset_manager import DatasetManager
from services.training.gpu_manager import GPUManager
from services.training.pipelines import PipelineFactory
from services.training.types import TrainingJobConfig, TrainingResult


@dataclass
class TrainingJobStatus:
    """Training job status information"""

    job_id: str
    status: str
    progress: float = 0.0
    error: Optional[str] = None


class TrainingService(BaseService):
    """
    Training Service - ML model training and management

    Features:
    - Distributed training support
    - Automatic hyperparameter tuning
    - Model versioning and registry
    - GPU acceleration
    - Experiment tracking
    """

    def __init__(self):
        super().__init__("training")
        self._model_registry: Optional[ModelRegistry] = None
        self._dataset_manager: Optional[DatasetManager] = None
        self._gpu_manager: Optional[GPUManager] = None
        self._training_queue: Optional[Space] = None
        self._active_jobs: Dict[str, Any] = {}
        self._job_statuses: Dict[str, TrainingJobStatus] = {}

    async def initialize(self) -> None:
        """Initialize training service with proper error handling"""
        try:
            await self._setup_mlflow()
            await self._setup_model_registry()
            await self._setup_gpu_manager()
            await self._setup_training_infrastructure()

            self.logger.info("Training service initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize training service: {e}")
            raise

    async def _setup_mlflow(self) -> None:
        """Setup MLflow experiment tracking"""
        mlflow_config = self.config.get("mlflow", {})

        if tracking_uri := mlflow_config.get("tracking_uri"):
            mlflow.set_tracking_uri(tracking_uri)

        if experiment_name := mlflow_config.get("experiment_name"):
            mlflow.set_experiment(experiment_name)

        self.logger.info("MLflow tracking configured")

    async def _setup_model_registry(self) -> None:
        """Initialize model registry"""
        registry_config = self.config.get("model_registry", {})
        self._model_registry = ModelRegistry(registry_config)
        await self._model_registry.initialize()
        self.logger.info("Model registry initialized")

    async def _setup_gpu_manager(self) -> None:
        """Setup GPU manager if GPUs are available"""
        if torch.cuda.is_available():
            gpu_config = self.config.get("gpu", {})
            self._gpu_manager = GPUManager(gpu_config)
            device_count = torch.cuda.device_count()
            device_name = (
                torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            )
            self.logger.info(
                f"GPU manager initialized - {device_count} GPUs available ({device_name})"
            )
        else:
            self.logger.info("No GPU available, using CPU for training")

    async def _setup_training_infrastructure(self) -> None:
        """Setup training queue and dataset manager"""
        # Initialize training queue
        self._training_queue = Space()

        # Initialize dataset manager
        dataset_config = self.config.get("datasets", {})
        self._dataset_manager = DatasetManager(dataset_config)
        await self._dataset_manager.initialize()

        self.logger.info("Training infrastructure setup complete")

    async def train_model(self, job_config: TrainingJobConfig) -> TrainingResult:
        """
        Execute a training job with comprehensive error handling

        Args:
            job_config: Configuration for the training job

        Returns:
            TrainingResult with job details and metrics
        """
        job_id = self._generate_job_id()

        # Initialize job status
        self._job_statuses[job_id] = TrainingJobStatus(
            job_id=job_id, status="initializing"
        )

        try:
            # Validate configuration
            await self._validate_job_config(job_config)

            # Create and configure pipeline
            pipeline = await self._create_pipeline(job_config)
            self._active_jobs[job_id] = pipeline

            # Execute training with MLflow tracking
            async with self._mlflow_context(job_config, job_id):
                result = await self._execute_training(pipeline, job_config, job_id)

            self._job_statuses[job_id].status = "completed"
            return result

        except Exception as e:
            self.logger.error(f"Training job {job_id} failed: {e}")
            self._job_statuses[job_id].status = "failed"
            self._job_statuses[job_id].error = str(e)
            raise

        finally:
            # Cleanup
            self._active_jobs.pop(job_id, None)

    async def _validate_job_config(self, job_config: TrainingJobConfig) -> None:
        """Validate training job configuration"""
        if not job_config.model_name:
            raise ValueError("Model name is required")

        if not job_config.pipeline_type:
            raise ValueError("Pipeline type is required")

        # Additional validation can be added here
        self.logger.debug(f"Job configuration validated for {job_config.model_name}")

    async def _create_pipeline(self, job_config: TrainingJobConfig) -> Any:
        """Create training pipeline"""
        pipeline = PipelineFactory.create(
            pipeline_type=job_config.pipeline_type,
            config=job_config,
            gpu_manager=self._gpu_manager,
            dataset_manager=self._dataset_manager,
        )

        self.logger.info(f"Created {job_config.pipeline_type} pipeline")
        return pipeline

    @asynccontextmanager
    async def _mlflow_context(self, job_config: TrainingJobConfig, job_id: str):
        """Context manager for MLflow run tracking"""
        run_name = f"{job_config.model_name}_{job_id}"

        with mlflow.start_run(run_name=run_name):
            # Log job parameters
            job_params = job_config.to_dict() if hasattr(job_config, "to_dict") else {}
            mlflow.log_params(job_params)
            mlflow.log_param("job_id", job_id)
            mlflow.log_param("model_name", job_config.model_name)
            mlflow.log_param("pipeline_type", job_config.pipeline_type)

            # Log system information
            import platform
            import torch
            mlflow.log_param("python_version", platform.python_version())
            mlflow.log_param("platform", platform.system())
            if torch.cuda.is_available():
                mlflow.log_param("gpu_count", torch.cuda.device_count())
                mlflow.log_param("gpu_name", torch.cuda.get_device_name(0))

            try:
                yield
            except Exception as e:
                mlflow.log_param("error", str(e))
                mlflow.log_metric("training_failed", 1)
                raise

    async def _execute_training(
        self, pipeline: Any, job_config: TrainingJobConfig, job_id: str
    ) -> TrainingResult:
        """Execute the actual training process"""
        self._job_statuses[job_id].status = "training"

        # Train model
        model = await pipeline.train()
        self._job_statuses[job_id].progress = 0.8

        # Evaluate model
        metrics = await pipeline.evaluate(model)
        
        # Log metrics with enhanced tracking
        mlflow.log_metrics(metrics)
        
        # Log additional trading-specific metrics if available
        if isinstance(metrics, dict):
            # Log Sharpe ratio if available
            if "sharpe_ratio" in metrics:
                mlflow.log_metric("sharpe_ratio", metrics["sharpe_ratio"])
            
            # Log return metrics
            if "total_return" in metrics:
                mlflow.log_metric("total_return", metrics["total_return"])
            if "annualized_return" in metrics:
                mlflow.log_metric("annualized_return", metrics["annualized_return"])
            
            # Log risk metrics
            if "max_drawdown" in metrics:
                mlflow.log_metric("max_drawdown", metrics["max_drawdown"])
            if "volatility" in metrics:
                mlflow.log_metric("volatility", metrics["volatility"])
            
            # Log accuracy/precision metrics for classification
            if "accuracy" in metrics:
                mlflow.log_metric("accuracy", metrics["accuracy"])
            if "precision" in metrics:
                mlflow.log_metric("precision", metrics["precision"])
            if "recall" in metrics:
                mlflow.log_metric("recall", metrics["recall"])
            if "f1_score" in metrics:
                mlflow.log_metric("f1_score", metrics["f1_score"])
            
            # Log regression metrics
            if "mse" in metrics:
                mlflow.log_metric("mse", metrics["mse"])
            if "rmse" in metrics:
                mlflow.log_metric("rmse", metrics["rmse"])
            if "mae" in metrics:
                mlflow.log_metric("mae", metrics["mae"])
            if "r2_score" in metrics:
                mlflow.log_metric("r2_score", metrics["r2_score"])
        
        self._job_statuses[job_id].progress = 0.9

        # Save model to registry
        model_info = await self._save_model(model, job_config, metrics)
        self._job_statuses[job_id].progress = 1.0

        return TrainingResult(
            job_id=job_id,
            model_id=model_info.model_id,
            metrics=metrics,
            status="completed",
        )

    async def _save_model(
        self, model: Any, job_config: TrainingJobConfig, metrics: Dict[str, Any]
    ) -> Any:
        """Save trained model to registry"""
        if not self._model_registry:
            raise RuntimeError("Model registry not initialized")

        model_info = await self._model_registry.save_model(
            model=model,
            name=job_config.model_name,
            version=job_config.version or "latest",
            metrics=metrics,
            metadata=job_config.metadata or {},
        )

        # Log model to MLflow
        mlflow.log_param("model_id", model_info.model_id)

        self.logger.info(f"Model saved with ID: {model_info.model_id}")
        return model_info

    def _generate_job_id(self) -> str:
        """Generate unique job ID"""
        return str(uuid.uuid4())

    async def get_job_status(self, job_id: str) -> Optional[TrainingJobStatus]:
        """Get status of a training job"""
        return self._job_statuses.get(job_id)

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running training job"""
        if job_id in self._active_jobs:
            pipeline = self._active_jobs[job_id]
            if hasattr(pipeline, "cancel"):
                await pipeline.cancel()

            self._job_statuses[job_id].status = "cancelled"
            del self._active_jobs[job_id]

            self.logger.info(f"Training job {job_id} cancelled")
            return True

        return False

    async def list_active_jobs(self) -> Dict[str, TrainingJobStatus]:
        """List all active training jobs"""
        return {
            job_id: status
            for job_id, status in self._job_statuses.items()
            if status.status in ["initializing", "training"]
        }

    async def cleanup(self) -> None:
        """Cleanup resources on service shutdown"""
        # Cancel all active jobs
        for job_id in list(self._active_jobs.keys()):
            await self.cancel_job(job_id)

        # Cleanup model registry
        if self._model_registry:
            await self._model_registry.cleanup()

        # Cleanup dataset manager
        if self._dataset_manager:
            await self._dataset_manager.cleanup()

        self.logger.info("Training service cleanup completed")
