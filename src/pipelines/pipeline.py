from framework.base import BaseComponent


class TrainingPipeline(BaseComponent):
    def __init__(self, config: dict):
        super().__init__("training_pipeline", config)
        self.stages = []

    async def execute(self, job_config: dict):
        """Execute training pipeline"""
        # Data preparation
        dataset = await self._prepare_data(job_config)

        # Feature engineering
        features = await self._engineer_features(dataset)

        # Model training
        model = await self._train_model(features, job_config)

        # Model evaluation
        metrics = await self._evaluate_model(model, features)

        # Model registration
        if metrics["performance"] > job_config["min_performance"]:
            await self._register_model(model, metrics)

        return {"model_id": model.id, "metrics": metrics, "status": "completed"}

    async def _prepare_data(self, job_config):
        """Prepare training data"""
        # Load raw data
        raw_data = await self.dataset_manager.load(
            symbols=job_config["symbols"],
            start_date=job_config["start_date"],
            end_date=job_config["end_date"],
        )

        # Clean and validate
        cleaned_data = await self._clean_data(raw_data)

        # Create features and labels
        features, labels = await self._create_features_labels(
            cleaned_data, job_config["prediction_horizon"]
        )

        # Split data
        splits = await self._split_data(
            features, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        )

        return splits
