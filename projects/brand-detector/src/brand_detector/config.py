"""Brand detector configuration."""

import os
from dataclasses import dataclass, field

from ml_shared.config_base import env_bool, env_float, env_int


@dataclass
class Config:
    # Data
    data_dir: str = field(default_factory=lambda: os.getenv("TRAIN_DATA_DIR", "data"))
    model_output_dir: str = "model/best_model"
    last_checkpoint_dir: str = "model/last_checkpoint"
    train_file: str = "brands_train.csv"
    val_split: float = 0.1

    # Model
    model_name: str = field(default_factory=lambda: os.getenv("TRAIN_MODEL_NAME", "prajjwal1/bert-tiny"))
    tokenizer_name: str = "bert-base-uncased"
    max_length: int = field(default_factory=lambda: env_int("TRAIN_MAX_LENGTH", 64))

    # Optimisation
    batch_size: int = field(default_factory=lambda: env_int("TRAIN_BATCH_SIZE", 64))
    eval_batch_size: int = field(default_factory=lambda: env_int("TRAIN_EVAL_BATCH_SIZE", 128))
    learning_rate: float = field(default_factory=lambda: env_float("TRAIN_LR", 2e-5))
    num_epochs: int = field(default_factory=lambda: env_int("TRAIN_EPOCHS", 5))
    use_amp: bool = field(default_factory=lambda: env_bool("TRAIN_USE_AMP", True))

    # Serving
    confidence_threshold: float = field(default_factory=lambda: env_float("SERVE_CONFIDENCE_THRESHOLD", 0.70))
    db_dir: str = field(default_factory=lambda: os.getenv("DB_DIR", "db"))

    # Experiment tracking
    mlflow_tracking_uri: str = field(default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", "mlruns"))
    mlflow_experiment: str = "brand-detector"
    mlflow_model_name: str = field(default_factory=lambda: os.getenv("MLFLOW_MODEL_NAME", ""))
    mlflow_model_stage: str = field(default_factory=lambda: os.getenv("MLFLOW_MODEL_STAGE", "Production"))
