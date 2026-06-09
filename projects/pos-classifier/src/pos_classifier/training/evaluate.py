"""Evaluation helpers: metrics, confusion matrix, classification report."""

import io
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

from pos_classifier.config import ID_TO_LABEL, NUM_LABELS

logger = logging.getLogger(__name__)


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Run inference on a DataLoader and return metrics dict."""
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    label_names = [ID_TO_LABEL[i] for i in range(NUM_LABELS)]
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    per_class_f1 = f1_score(
        all_labels, all_preds, average=None, labels=list(range(NUM_LABELS)), zero_division=0
    )
    report = classification_report(
        all_labels, all_preds, target_names=label_names, zero_division=0
    )

    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_f1": {
            ID_TO_LABEL[i]: float(per_class_f1[i]) for i in range(NUM_LABELS)
        },
        "classification_report": report,
    }
    logger.info("Accuracy: %.4f  Macro-F1: %.4f", acc, macro_f1)
    logger.info("\n%s", report)
    return metrics


def save_confusion_matrix(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_path: Path,
) -> None:
    """Generate and save a confusion matrix PNG."""
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    label_names = [ID_TO_LABEL[i] for i in range(NUM_LABELS)]
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_LABELS)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalised Confusion Matrix")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", output_path)
