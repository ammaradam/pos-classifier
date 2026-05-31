"""MLflow Model Registry management: registration and stage transitions."""

import logging
from pathlib import Path

import mlflow
import mlflow.pytorch
from mlflow.entities import ModelVersion

logger = logging.getLogger(__name__)


def register_model(
    model_path: Path,
    run_id: str,
    model_name: str = "pos-classifier",
    description: str = "",
) -> ModelVersion:
    """
    Register a model artifact from an MLflow run to the Model Registry.

    Args:
        model_path: Local path to saved model (will be uploaded as artifact).
        run_id: MLflow run ID to register from.
        model_name: Name for the registered model (default: "pos-classifier").
        description: Optional description of the model version.

    Returns:
        MLflow ModelVersion object with version number and URI.

    Raises:
        ValueError: If run_id doesn't exist or model_path is invalid.
    """
    if not model_path.exists():
        raise ValueError(f"Model path does not exist: {model_path}")

    # Ensure model exists in MLflow backend
    run = mlflow.get_run(run_id)
    if run is None:
        raise ValueError(f"Run {run_id} not found in MLflow tracking server")

    # Register model from artifact
    model_uri = f"runs://{run_id}/model_artifact"
    try:
        version = mlflow.register_model(model_uri, model_name)
        logger.info(
            "Model registered: %s v%s (run_id=%s)",
            model_name,
            version.version,
            run_id,
        )
        if description:
            client = mlflow.tracking.MlflowClient()
            client.update_model_version(model_name, version.version, description)
            logger.info("Model description updated: %s", description)
        return version
    except Exception as e:
        logger.error("Failed to register model: %s", e)
        raise


def transition_model_stage(
    model_name: str,
    version: int | str,
    stage: str,
) -> ModelVersion:
    """
    Transition a model version to a new stage (Staging, Production, Archived).

    Args:
        model_name: Name of the registered model.
        version: Version number (int or str).
        stage: Target stage: "Staging", "Production", or "Archived".

    Returns:
        Updated MLflow ModelVersion object.

    Raises:
        ValueError: If stage is invalid or version doesn't exist.
    """
    valid_stages = {"Staging", "Production", "Archived", "None"}
    if stage not in valid_stages:
        raise ValueError(f"Invalid stage '{stage}'. Must be one of {valid_stages}")

    try:
        client = mlflow.tracking.MlflowClient()
        version_obj = client.transition_model_version_stage(
            model_name, str(version), stage
        )
        logger.info(
            "Model transitioned: %s v%s → %s",
            model_name,
            version,
            stage,
        )
        return version_obj
    except Exception as e:
        logger.error("Failed to transition model: %s", e)
        raise


def get_model_version(model_name: str, version: int | str) -> ModelVersion:
    """Get metadata for a specific model version."""
    client = mlflow.tracking.MlflowClient()
    return client.get_model_version(model_name, str(version))


def list_model_versions(model_name: str) -> list[ModelVersion]:
    """List all versions of a registered model."""
    client = mlflow.tracking.MlflowClient()
    return client.search_model_versions(f"name='{model_name}'")
