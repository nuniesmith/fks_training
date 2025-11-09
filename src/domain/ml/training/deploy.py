"""
Model Deployment Script

Handles model promotion through stages: development → staging → production
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import mlflow
from mlflow.tracking import MlflowClient

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))


def promote_model(
    model_name: str, version: str, target_stage: str, tracking_uri: Optional[str] = None
) -> None:
    """
    Promote a model version to a target stage.

    Args:
        model_name: Name of the model
        version: Model version to promote
        target_stage: Target stage (staging, production, archive, None)
        tracking_uri: MLflow tracking URI (optional)
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()

    # Transition model version to target stage
    client.transition_model_version_stage(
        name=model_name, version=version, stage=target_stage
    )

    print(f"Model {model_name} version {version} promoted to {target_stage}")


def list_models_by_stage(
    model_name: str, stage: str, tracking_uri: Optional[str] = None
) -> list:
    """
    List model versions in a specific stage.

    Args:
        model_name: Name of the model
        stage: Stage to filter by
        tracking_uri: MLflow tracking URI (optional)

    Returns:
        List of model versions
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()
    versions = client.get_latest_versions(model_name, stages=[stage])

    return versions


def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="Deploy ML models through stages")
    parser.add_argument("--model-uri", type=str, required=True, help="Model URI (e.g., runs:/run_id/model)")
    parser.add_argument("--model-name", type=str, help="Model name in registry")
    parser.add_argument("--stage", type=str, default="staging", choices=["staging", "production", "archive", "None"])
    parser.add_argument("--tracking-uri", type=str, help="MLflow tracking URI")
    parser.add_argument("--action", type=str, default="promote", choices=["promote", "list", "register"])

    args = parser.parse_args()

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)

    client = MlflowClient()

    if args.action == "register":
        # Register model from run
        if not args.model_name:
            raise ValueError("--model-name required for register action")

        # Extract run_id from model_uri if it's a runs:/ URI
        if args.model_uri.startswith("runs:/"):
            run_id = args.model_uri.split("/")[1]
            model_path = args.model_uri.split("/", 2)[2] if "/" in args.model_uri.split("/", 2)[2:] else "model"
        else:
            raise ValueError("Model URI must be in format runs:/run_id/model")

        # Register model
        model_version = mlflow.register_model(args.model_uri, args.model_name)
        print(f"Registered model {args.model_name} version {model_version.version}")

        # Transition to target stage
        if args.stage != "None":
            client.transition_model_version_stage(
                name=args.model_name, version=model_version.version, stage=args.stage
            )
            print(f"Transitioned to {args.stage} stage")

    elif args.action == "promote":
        if not args.model_name:
            raise ValueError("--model-name required for promote action")

        # Extract version from model_uri or use latest
        if "/" in args.model_uri:
            # Assume format models:/name/version or runs:/run_id/model
            if args.model_uri.startswith("models:/"):
                parts = args.model_uri.replace("models:/", "").split("/")
                model_name_from_uri = parts[0]
                version = parts[1] if len(parts) > 1 else None
            else:
                version = None
        else:
            version = args.model_uri

        if not version:
            # Get latest version
            versions = client.get_latest_versions(args.model_name, stages=[])
            if not versions:
                raise ValueError(f"No versions found for model {args.model_name}")
            version = versions[0].version

        promote_model(args.model_name, version, args.stage, args.tracking_uri)

    elif args.action == "list":
        if not args.model_name:
            raise ValueError("--model-name required for list action")

        versions = list_models_by_stage(args.model_name, args.stage, args.tracking_uri)
        print(f"\nModels in {args.stage} stage:")
        for v in versions:
            print(f"  Version {v.version}: {v.current_stage} (Run: {v.run_id})")


if __name__ == "__main__":
    main()

