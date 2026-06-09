"""Training and serving configuration."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ml_shared.config_base import env_bool as _env_bool
from ml_shared.config_base import env_float as _env_float
from ml_shared.config_base import env_int as _env_int


LABEL_MAP: dict[str, int] = {
    "Dry Goods & Pantry Staples": 0,
    "Fresh & Perishable Items": 1,
    "Household & Personal Care": 2,
    "Beverages": 3,
    "Specialty & Miscellaneous": 4,
}
ID_TO_LABEL: dict[int, str] = {v: k for k, v in LABEL_MAP.items()}
NUM_LABELS: int = len(LABEL_MAP)


@dataclass
class TrainingConfig:
    # Data
    data_dir: str = field(default_factory=lambda: os.getenv("TRAIN_DATA_DIR", "data"))
    model_output_dir: str = "model/best_model"
    last_checkpoint_dir: str = "model/last_checkpoint"
    train_file: str = "Training_data.csv"
    val_split: float = 0.1
    test_split: float = 0.1

    # Model
    model_name: str = field(default_factory=lambda: os.getenv("TRAIN_MODEL_NAME", "prajjwal1/bert-tiny"))
    # prajjwal1/bert-tiny ships model weights only — no tokenizer files.
    # It uses the identical WordPiece vocabulary as bert-base-uncased, so we
    # load the tokenizer from there and save it alongside the fine-tuned weights.
    tokenizer_name: str = "bert-base-uncased"
    model_revision: Optional[str] = None
    # p99 token count in training data is ~20; 64 covers 100% with 3× headroom
    # and halves BERT's O(n²) attention cost vs the previous 128 default.
    max_length: int = field(default_factory=lambda: _env_int("TRAIN_MAX_LENGTH", 64))

    # Optimisation
    batch_size: int = field(default_factory=lambda: _env_int("TRAIN_BATCH_SIZE", 64))
    eval_batch_size: int = field(default_factory=lambda: _env_int("TRAIN_EVAL_BATCH_SIZE", 128))
    learning_rate: float = field(default_factory=lambda: _env_float("TRAIN_LR", 2e-5))
    weight_decay: float = field(default_factory=lambda: _env_float("TRAIN_WEIGHT_DECAY", 0.01))
    num_epochs: int = field(default_factory=lambda: _env_int("TRAIN_EPOCHS", 5))
    warmup_ratio: float = field(default_factory=lambda: _env_float("TRAIN_WARMUP_RATIO", 0.1))
    early_stopping_patience: int = field(default_factory=lambda: _env_int("TRAIN_PATIENCE", 2))
    gradient_accumulation_steps: int = field(default_factory=lambda: _env_int("TRAIN_GRAD_ACCUM", 1))

    # GPU optimisations (no-ops on CPU)
    use_amp: bool = field(default_factory=lambda: _env_bool("TRAIN_USE_AMP", True))
    # num_workers > 0 speeds up DataLoader on Linux/Mac; keep 0 on Windows to avoid
    # multiprocessing spawn issues unless you run inside WSL2
    num_workers: int = field(default_factory=lambda: _env_int("TRAIN_NUM_WORKERS", 0))

    # Serving
    confidence_threshold: float = field(default_factory=lambda: _env_float("SERVE_CONFIDENCE_THRESHOLD", 0.70))
    review_sample_rate: float = 0.20

    # Runtime storage (separate from model artefacts so volumes don't shadow baked model)
    db_dir: str = field(default_factory=lambda: os.getenv("DB_DIR", "db"))

    # Experiment tracking
    mlflow_tracking_uri: str = field(default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", "mlruns"))
    mlflow_experiment: str = "pos-classifier"
    # Registry: if set, Predictor loads from MLflow registry instead of local path
    mlflow_model_name: str = field(default_factory=lambda: os.getenv("MLFLOW_MODEL_NAME", ""))
    mlflow_model_stage: str = field(default_factory=lambda: os.getenv("MLFLOW_MODEL_STAGE", "Production"))

    # Subset for quick smoke-test (0 = use all data)
    data_subset: int = field(default_factory=lambda: _env_int("TRAIN_SUBSET", 0))

    def model_output_path(self) -> Path:
        return Path(self.model_output_dir)

    def last_checkpoint_path(self) -> Path:
        return Path(self.last_checkpoint_dir)
