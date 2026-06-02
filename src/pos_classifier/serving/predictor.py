"""Model loading and batch inference with confidence scoring."""

import json
import logging
import random
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from pos_classifier.config import ID_TO_LABEL, LABEL_MAP, NUM_LABELS
from pos_classifier.serving.schema import PredictResponse

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_DIR = Path("model/best_model")
_DEFAULT_DB_PATH = Path("db/predictions.db")


class Predictor:
    """
    Singleton-style wrapper around the fine-tuned model.

    Call ``Predictor.get()`` to obtain the shared instance (lazy-loaded).
    """

    _instance: Optional["Predictor"] = None

    def __init__(
        self,
        model_dir: Path = _DEFAULT_MODEL_DIR,
        db_path: Path = _DEFAULT_DB_PATH,
        mlflow_model_name: str = "",
        mlflow_model_stage: str = "Production",
    ) -> None:
        self.model_dir = model_dir
        self.db_path = db_path
        self.mlflow_model_name = mlflow_model_name
        self.mlflow_model_stage = mlflow_model_stage
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()
        self._init_db()

    def _load_model(self) -> None:
        if self.mlflow_model_name:
            self._load_from_registry()
        else:
            self._load_from_local()

    def _load_from_registry(self) -> None:
        import mlflow
        model_uri = f"models:/{self.mlflow_model_name}/{self.mlflow_model_stage}"
        logger.info("Loading model from MLflow registry: %s", model_uri)
        local_path = mlflow.artifacts.download_artifacts(model_uri)
        self.model_dir = Path(local_path)
        self._load_from_local()

    def _load_from_local(self) -> None:
        logger.info("Loading model from %s …", self.model_dir)
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir.resolve()}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_dir)
        ).to(self.device)
        self.model.eval()

        meta_path = self.model_dir / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self.model_version: str = meta.get("model_version", "unknown")
            self.confidence_threshold: float = meta.get("confidence_threshold", 0.70)
            self.review_sample_rate: float = 0.20
        else:
            self.model_version = "unknown"
            self.confidence_threshold = 0.70
            self.review_sample_rate = 0.20

        logger.info("Model loaded. version=%s  threshold=%.2f", self.model_version, self.confidence_threshold)

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                product_description TEXT    NOT NULL,
                predicted_category  TEXT    NOT NULL,
                confidence          REAL    NOT NULL,
                flagged_for_review  INTEGER NOT NULL,
                model_version       TEXT    NOT NULL,
                predicted_at        TEXT    NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                product_description  TEXT NOT NULL,
                corrected_category   TEXT NOT NULL,
                original_prediction  TEXT,
                submitted_at         TEXT NOT NULL
            )
            """
        )
        conn.commit()
        conn.close()

    @classmethod
    def get(
        cls,
        model_dir: Path = _DEFAULT_MODEL_DIR,
        db_path: Path = _DEFAULT_DB_PATH,
        mlflow_model_name: str = "",
        mlflow_model_stage: str = "Production",
    ) -> "Predictor":
        if cls._instance is None:
            cls._instance = cls(
                model_dir=model_dir,
                db_path=db_path,
                mlflow_model_name=mlflow_model_name,
                mlflow_model_stage=mlflow_model_stage,
            )
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Force re-load on next ``get()`` call (used after retraining)."""
        cls._instance = None

    def predict(self, texts: list[str]) -> list[PredictResponse]:
        """Run batch inference and persist results to SQLite."""
        if not texts:
            return []

        now = datetime.now(timezone.utc)
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=64,
            return_tensors="pt",
        )
        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits

        probs = F.softmax(logits, dim=-1)
        confs, pred_ids = probs.max(dim=-1)

        results: list[PredictResponse] = []
        rows_to_insert: list[tuple] = []

        for text, conf, pred_id in zip(texts, confs.cpu().tolist(), pred_ids.cpu().tolist()):
            category = ID_TO_LABEL[pred_id]
            # Flag if below confidence threshold OR randomly sampled for human review
            flagged = conf < self.confidence_threshold or random.random() < self.review_sample_rate
            resp = PredictResponse(
                product_description=text,
                category=category,
                confidence=round(conf, 4),
                flagged_for_human_review=bool(flagged),
                model_version=self.model_version,
                predicted_at=now,
            )
            results.append(resp)
            rows_to_insert.append(
                (text, category, round(conf, 4), int(flagged), self.model_version, now.isoformat())
            )

        self._persist(rows_to_insert)
        return results

    def _persist(self, rows: list[tuple]) -> None:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                conn = sqlite3.connect(self.db_path, timeout=5.0)
                conn.execute("PRAGMA journal_mode=WAL")
                conn.executemany(
                    """INSERT INTO predictions
                       (product_description, predicted_category, confidence,
                        flagged_for_review, model_version, predicted_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    rows,
                )
                conn.commit()
                conn.close()
                return
            except sqlite3.OperationalError as e:
                conn.close() if 'conn' in locals() else None
                if attempt < max_retries - 1:
                    wait = 0.1 * (2 ** attempt)
                    logger.warning("DB write retry %d/%d after %.2fs (locked: %s)", attempt + 1, max_retries, wait, str(e))
                    time.sleep(wait)
                else:
                    logger.error("DB write failed after %d retries: %s", max_retries, e)
                    raise
            except Exception as e:
                logger.error("Unexpected DB error: %s", e)
                raise

    def record_feedback(self, description: str, corrected: str, original: Optional[str]) -> None:
        if corrected not in LABEL_MAP:
            raise ValueError(f"Invalid category '{corrected}'. Valid: {list(LABEL_MAP.keys())}")
        now = datetime.now(timezone.utc).isoformat()
        max_retries = 3
        for attempt in range(max_retries):
            try:
                conn = sqlite3.connect(self.db_path, timeout=5.0)
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute(
                    """INSERT INTO feedback
                       (product_description, corrected_category, original_prediction, submitted_at)
                       VALUES (?, ?, ?, ?)""",
                    (description, corrected, original, now),
                )
                conn.commit()
                conn.close()
                return
            except sqlite3.OperationalError as e:
                conn.close() if 'conn' in locals() else None
                if attempt < max_retries - 1:
                    wait = 0.1 * (2 ** attempt)
                    logger.warning("Feedback DB write retry %d/%d after %.2fs (locked: %s)", attempt + 1, max_retries, wait, str(e))
                    time.sleep(wait)
                else:
                    logger.error("Feedback DB write failed after %d retries: %s", max_retries, e)
                    raise
            except Exception as e:
                logger.error("Unexpected feedback DB error: %s", e)
                raise
