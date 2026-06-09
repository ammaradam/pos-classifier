"""MLflow Model Registry helpers shared across projects."""

import logging
from pathlib import Path

import mlflow
from mlflow.entities import ModelVersion

logger = logging.getLogger(__name__)


def register_model(
    model_path: Path,
    run_id: str,
    model_name: str,
    description: str = "",
) -> ModelVersion:
    if not model_path.exists():
        raise ValueError(f"Model path does not exist: {model_path}")

    run = mlflow.get_run(run_id)
    if run is None:
        raise ValueError(f"Run {run_id} not found in MLflow tracking server")

    model_uri = f"runs:/{run_id}/model_artifact"
    try:
        version = mlflow.register_model(model_uri, model_name)
        logger.info("Model registered: %s v%s (run_id=%s)", model_name, version.version, run_id)
        if description:
            client = mlflow.tracking.MlflowClient()
            client.update_model_version(model_name, version.version, description)
        return version
    except Exception as e:
        logger.error("Failed to register model: %s", e)
        raise


def transition_model_stage(model_name: str, version: int | str, stage: str) -> ModelVersion:
    valid_stages = {"Staging", "Production", "Archived", "None"}
    if stage not in valid_stages:
        raise ValueError(f"Invalid stage '{stage}'. Must be one of {valid_stages}")
    try:
        client = mlflow.tracking.MlflowClient()
        version_obj = client.transition_model_version_stage(model_name, str(version), stage)
        logger.info("Model transitioned: %s v%s → %s", model_name, version, stage)
        return version_obj
    except Exception as e:
        logger.error("Failed to transition model: %s", e)
        raise
