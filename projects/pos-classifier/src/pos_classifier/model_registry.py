"""MLflow Model Registry management — delegates to ml_shared.mlflow_utils."""

from pathlib import Path

import mlflow
from mlflow.entities import ModelVersion
from mlflow.tracking import MlflowClient

from ml_shared.mlflow_utils import register_model as _register, transition_model_stage as _transition


def register_model(
    model_path: Path,
    run_id: str,
    model_name: str = "pos-classifier",
    description: str = "",
) -> ModelVersion:
    return _register(model_path, run_id, model_name, description)


def transition_model_stage(model_name: str, version: int | str, stage: str) -> ModelVersion:
    return _transition(model_name, version, stage)


def get_model_version(model_name: str, version: int | str) -> ModelVersion:
    return MlflowClient().get_model_version(model_name, str(version))


def list_model_versions(model_name: str) -> list[ModelVersion]:
    return MlflowClient().search_model_versions(f"name='{model_name}'")
