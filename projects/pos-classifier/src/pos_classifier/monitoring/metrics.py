"""KPI computation: compare model predictions against human-verified labels."""

import logging
import sqlite3
from pathlib import Path
from typing import Optional

from pos_classifier.config import ID_TO_LABEL, LABEL_MAP, NUM_LABELS
from pos_classifier.data.preprocessing import load_query_data
from pos_classifier.serving.predictor import Predictor

logger = logging.getLogger(__name__)


def _fetch_predictions(db_path: Path) -> list[dict]:
    """Return all rows from the predictions table."""
    if not db_path.exists():
        return []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY predicted_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _fetch_feedback(db_path: Path) -> list[dict]:
    if not db_path.exists():
        return []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM feedback ORDER BY submitted_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def compute_validation_metrics(
    query_csv: Path,
    predictor: Predictor,
) -> dict:
    """
    Run inference on all query rows and compare against human-verified labels
    for the ~1,166 rows that have them.

    Returns a metrics dict consumed by the dashboard.
    """
    texts, gt_labels = load_query_data(query_csv)
    logger.info("Running inference on %d query rows …", len(texts))

    # Batch in chunks to avoid OOM on large inputs
    chunk_size = 256
    all_preds: list[dict] = []
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i : i + chunk_size]
        all_preds.extend(predictor.predict(chunk))

    # Only evaluate rows that have human-verified labels
    verified_pairs: list[tuple[str, str]] = []
    for pred, gt in zip(all_preds, gt_labels):
        if gt is not None:
            verified_pairs.append((pred.category, ID_TO_LABEL[gt]))

    if not verified_pairs:
        logger.warning("No verified labels found in query data.")
        return {"error": "no verified labels"}

    correct = sum(1 for p, g in verified_pairs if p == g)
    total = len(verified_pairs)
    overall_acc = correct / total

    # Per-class accuracy
    per_class: dict[str, dict] = {
        name: {"correct": 0, "total": 0} for name in ID_TO_LABEL.values()
    }
    for pred_cat, gt_cat in verified_pairs:
        per_class[gt_cat]["total"] += 1
        if pred_cat == gt_cat:
            per_class[gt_cat]["correct"] += 1

    per_class_acc = {
        name: (
            round(d["correct"] / d["total"], 4) if d["total"] > 0 else None
        )
        for name, d in per_class.items()
    }

    # Prediction distribution across all query rows
    dist: dict[str, int] = {name: 0 for name in ID_TO_LABEL.values()}
    for p in all_preds:
        dist[p.category] += 1

    # Confidence stats
    confidences = [p.confidence for p in all_preds]
    flagged_count = sum(1 for p in all_preds if p.flagged_for_human_review)

    metrics = {
        "overall_accuracy": round(overall_acc, 4),
        "verified_count": total,
        "per_class_accuracy": per_class_acc,
        "prediction_distribution": dist,
        "total_predicted": len(all_preds),
        "flagged_count": flagged_count,
        "avg_confidence": round(sum(confidences) / len(confidences), 4),
        "low_confidence_rate": round(
            sum(1 for c in confidences if c < predictor.confidence_threshold) / len(confidences),
            4,
        ),
        "model_version": predictor.model_version,
    }
    logger.info(
        "Validation metrics: accuracy=%.4f  verified=%d",
        overall_acc,
        total,
    )
    return metrics


def get_db_summary(db_path: Path) -> dict:
    """Return lightweight summary stats from the predictions DB."""
    preds = _fetch_predictions(db_path)
    feedback = _fetch_feedback(db_path)
    if not preds:
        return {"total_predictions": 0, "total_feedback": len(feedback)}

    dist: dict[str, int] = {}
    flagged = 0
    for p in preds:
        cat = p["predicted_category"]
        dist[cat] = dist.get(cat, 0) + 1
        flagged += p["flagged_for_review"]

    return {
        "total_predictions": len(preds),
        "total_feedback": len(feedback),
        "flagged_count": flagged,
        "category_distribution": dist,
        "latest_prediction": preds[0]["predicted_at"] if preds else None,
    }


def pending_retraining_check(db_path: Path, threshold: int = 500) -> bool:
    """Return True when enough feedback has accumulated to trigger retraining."""
    feedback = _fetch_feedback(db_path)
    return len(feedback) >= threshold
