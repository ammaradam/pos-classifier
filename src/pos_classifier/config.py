"""Training and serving configuration."""

from dataclasses import dataclass, field
from pathlib import Path


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
    data_dir: str = "data"
    model_output_dir: str = "model/best_model"
    last_checkpoint_dir: str = "model/last_checkpoint"
    train_file: str = "Training_data.csv"
    val_split: float = 0.1
    test_split: float = 0.1

    # Model
    model_name: str = "prajjwal1/bert-tiny"
    # refs/pr/13 adds model.safetensors to bert-tiny, avoiding the torch>=2.6
    # requirement that transformers 5.x enforces for legacy .bin files (CVE-2025-32434).
    # Set to "" or "main" once torch>=2.6 is installed.
    # Empty string loads from the default "main" branch.
    # Previously pinned to "refs/pr/13" (safetensors) as a workaround for the
    # torch<2.6 .bin restriction (CVE-2025-32434). No longer needed with torch>=2.6.
    model_revision: str = ""
    # prajjwal1/bert-tiny ships model weights only — no tokenizer files.
    # It uses the identical WordPiece vocabulary as bert-base-uncased, so we
    # load the tokenizer from there and save it alongside the fine-tuned weights.
    tokenizer_name: str = "bert-base-uncased"
    # p99 token count in training data is ~20; 64 covers 100% with 3× headroom
    # and halves BERT's O(n²) attention cost vs the previous 128 default.
    max_length: int = 64

    # Optimisation
    batch_size: int = 64          # GPU: safe to double/quadruple (128-256 for 6GB+ VRAM)
    eval_batch_size: int = 128
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 5
    warmup_ratio: float = 0.1
    early_stopping_patience: int = 2
    gradient_accumulation_steps: int = 1  # simulate larger batch on low-VRAM GPUs

    # GPU optimisations (no-ops on CPU)
    use_amp: bool = True      # automatic mixed precision (FP16) — ~2x speedup on GPU
    # num_workers > 0 speeds up DataLoader on Linux/Mac; keep 0 on Windows to avoid
    # multiprocessing spawn issues unless you run inside WSL2
    num_workers: int = 0

    # Serving
    confidence_threshold: float = 0.70
    review_sample_rate: float = 0.20   # fraction of predictions sent for human review

    # Experiment tracking
    mlflow_tracking_uri: str = "mlruns"
    mlflow_experiment: str = "pos-classifier"

    # Subset for quick smoke-test (0 = use all data)
    data_subset: int = 0

    def model_output_path(self) -> Path:
        return Path(self.model_output_dir)

    def last_checkpoint_path(self) -> Path:
        return Path(self.last_checkpoint_dir)
