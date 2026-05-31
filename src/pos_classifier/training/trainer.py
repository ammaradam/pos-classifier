"""Fine-tuning loop with MLflow experiment tracking and early stopping."""

import copy
import datetime as dt
import json
import logging
import time
import uuid
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertConfig,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from pos_classifier.config import (
    ID_TO_LABEL,
    LABEL_MAP,
    NUM_LABELS,
    TrainingConfig,
)
from pos_classifier.data.preprocessing import POSDataset, load_training_data
from pos_classifier.training.evaluate import evaluate_model, save_confusion_matrix

logger = logging.getLogger(__name__)


def _save_best_checkpoint(model, tokenizer, out_dir: Path) -> None:
    """Save model + tokenizer in HuggingFace format for inference."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    logger.info("Best checkpoint saved → %s", out_dir)


def _save_last_checkpoint(
    model,
    tokenizer,
    optimizer: AdamW,
    scheduler,
    scaler: GradScaler,
    epoch: int,
    best_val_acc: float,
    no_improve: int,
    save_dir: Path,
) -> None:
    """Save full training state for resuming: model, tokenizer, and optimizer state."""
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    torch.save(
        {
            "epoch": epoch,
            "best_val_acc": best_val_acc,
            "no_improve": no_improve,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
        },
        save_dir / "training_state.pt",
    )
    logger.info("Last checkpoint saved (epoch %d) → %s", epoch, save_dir)


def _mlflow_key(name: str) -> str:
    """Sanitise a category name for use as an MLflow metric key.
    MLflow allows: alphanumerics, _, -, ., spaces, /
    """
    return name.replace("&", "and").replace(" ", "_")


def train(cfg: TrainingConfig) -> dict:
    """
    Fine-tune BERT-tiny for POS product classification.

    GPU optimisations enabled automatically when CUDA is available:
    - Automatic Mixed Precision (AMP/FP16)
    - pin_memory on DataLoaders for faster CPU→GPU transfer
    - cuDNN benchmark mode
    - Gradient accumulation (configurable)

    Logs all parameters and metrics to MLflow.
    Returns the final evaluation metrics dict from the test set.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    use_amp = cfg.use_amp and use_cuda  # AMP is a no-op on CPU, skip it
    logger.info(
        "Training device: %s  |  AMP: %s  |  num_workers: %d",
        device,
        use_amp,
        cfg.num_workers,
    )

    # cuDNN auto-tuner: finds the best convolution algorithm for fixed input shapes.
    # Gives ~5-15% speedup after the first batch on GPU; harmless on CPU.
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    # ── Data ──────────────────────────────────────────────────────────────────
    data_path = Path(cfg.data_dir) / cfg.train_file
    texts, labels = load_training_data(data_path, subset=cfg.data_subset)

    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels,
        test_size=cfg.val_split + cfg.test_split,
        stratify=labels,
        random_state=42,
    )
    relative_test = cfg.test_split / (cfg.val_split + cfg.test_split)
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels,
        test_size=relative_test,
        stratify=temp_labels,
        random_state=42,
    )
    logger.info(
        "Split: train=%d  val=%d  test=%d",
        len(train_texts), len(val_texts), len(test_texts),
    )

    # ── Tokenizer & Datasets ──────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    train_ds = POSDataset(train_texts, train_labels, tokenizer, cfg.max_length)
    val_ds = POSDataset(val_texts, val_labels, tokenizer, cfg.max_length)
    test_ds = POSDataset(test_texts, test_labels, tokenizer, cfg.max_length)

    # pin_memory=True pre-pages tensors into pinned (page-locked) CPU memory so
    # the CUDA DMA engine can transfer them to GPU without an extra CPU-side copy.
    # On CPU this flag is silently ignored.
    loader_kwargs = dict(pin_memory=use_cuda, num_workers=cfg.num_workers)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, **loader_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.eval_batch_size, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.eval_batch_size, **loader_kwargs)

    # ── Class weights (handles Specialty & Misc imbalance) ────────────────────
    cw = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(NUM_LABELS),
        y=np.array(train_labels),
    )
    class_weights = torch.tensor(cw, dtype=torch.float).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # ── Model ─────────────────────────────────────────────────────────────────
    # prajjwal1/bert-tiny predates the `model_type` field that AutoModel
    # requires. We load it explicitly as BERT — same architecture, no lookup.
    # model_revision pins to a PR that ships model.safetensors (required by
    # transformers 5.x on torch<2.6 due to CVE-2025-32434 .bin restriction).
    revision_kwargs = {"revision": cfg.model_revision} if cfg.model_revision else {}
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name,
            num_labels=NUM_LABELS,
            id2label=ID_TO_LABEL,
            label2id=LABEL_MAP,
            ignore_mismatched_sizes=True,
            **revision_kwargs,
        )
    except ValueError:
        logger.info("AutoModel failed (missing model_type); loading as BertForSequenceClassification.")
        bert_cfg = BertConfig.from_pretrained(cfg.model_name, **revision_kwargs)
        bert_cfg.num_labels = NUM_LABELS
        bert_cfg.id2label = ID_TO_LABEL
        bert_cfg.label2id = LABEL_MAP
        model = BertForSequenceClassification.from_pretrained(
            cfg.model_name,
            config=bert_cfg,
            ignore_mismatched_sizes=True,
            **revision_kwargs,
        )
    model = model.to(device)

    # ── AMP scaler ────────────────────────────────────────────────────────────
    # GradScaler prevents underflow when gradients are represented in FP16.
    # On CPU (use_amp=False) we use a dummy that does nothing.
    scaler = GradScaler("cuda", enabled=use_amp)

    # ── Optimiser & Scheduler ─────────────────────────────────────────────────
    # total_steps accounts for gradient accumulation: the optimiser steps every
    # `gradient_accumulation_steps` micro-batches, not every batch.
    effective_batches = len(train_loader) // cfg.gradient_accumulation_steps
    total_steps = effective_batches * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    # ── MLflow Run ────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment)

    out_dir = cfg.model_output_path()
    last_ckpt_dir = cfg.last_checkpoint_path()

    with mlflow.start_run():
        mlflow.log_params({k: v for k, v in cfg.__dict__.items() if not k.startswith("_")})

        best_val_acc = 0.0
        no_improve = 0
        best_state: dict | None = None
        current_epoch = 0

        try:
            for epoch in range(1, cfg.num_epochs + 1):
                current_epoch = epoch

                # ── Train epoch ───────────────────────────────────────────────
                model.train()
                total_loss = 0.0
                t0 = time.time()
                optimizer.zero_grad()

                for step, batch in enumerate(train_loader, start=1):
                    input_ids      = batch["input_ids"].to(device, non_blocking=True)
                    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                    lbl            = batch["labels"].to(device, non_blocking=True)

                    # autocast runs the forward pass in FP16 on GPU; no-op on CPU.
                    with autocast("cuda", enabled=use_amp):
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        loss = loss_fn(outputs.logits, lbl)
                        # Scale loss for gradient accumulation so effective LR is unchanged
                        loss = loss / cfg.gradient_accumulation_steps

                    scaler.scale(loss).backward()

                    if step % cfg.gradient_accumulation_steps == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        optimizer.zero_grad()

                    total_loss += loss.item() * cfg.gradient_accumulation_steps

                avg_train_loss = total_loss / len(train_loader)
                elapsed = time.time() - t0

                # ── Validate ──────────────────────────────────────────────────
                val_metrics = evaluate_model(model, val_loader, device)
                val_acc = val_metrics["accuracy"]
                val_f1  = val_metrics["macro_f1"]

                logger.info(
                    "Epoch %d/%d | train_loss=%.4f | val_acc=%.4f | val_macro_f1=%.4f | %.1fs",
                    epoch, cfg.num_epochs, avg_train_loss, val_acc, val_f1, elapsed,
                )
                mlflow.log_metrics(
                    {"train_loss": avg_train_loss, "val_accuracy": val_acc, "val_macro_f1": val_f1},
                    step=epoch,
                )

                # ── Best checkpoint ───────────────────────────────────────────
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    no_improve = 0
                    best_state = copy.deepcopy(model.state_dict())
                    _save_best_checkpoint(model, tokenizer, out_dir)
                else:
                    no_improve += 1

                # ── Last checkpoint (always, for resuming) ────────────────────
                _save_last_checkpoint(
                    model, tokenizer, optimizer, scheduler, scaler,
                    epoch, best_val_acc, no_improve, last_ckpt_dir,
                )

                # ── Early stopping ────────────────────────────────────────────
                if no_improve >= cfg.early_stopping_patience:
                    logger.info("Early stopping at epoch %d.", epoch)
                    break

            # ── Restore best weights for test evaluation ──────────────────────
            if best_state is not None:
                model.load_state_dict(best_state)

            # ── Final test evaluation ─────────────────────────────────────────
            test_metrics = evaluate_model(model, test_loader, device)

            # ── Persist best model again with tokenizer (no-op if already saved)
            out_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            logger.info("Best model saved to %s", out_dir)

            metadata = {
                "model_version": str(uuid.uuid4())[:8],
                "model_name": cfg.model_name,
                "accuracy": test_metrics["accuracy"],
                "macro_f1": test_metrics["macro_f1"],
                "per_class_f1": test_metrics["per_class_f1"],
                "trained_at": dt.datetime.utcnow().isoformat() + "Z",
                "data_rows_used": len(train_texts) + len(val_texts) + len(test_texts),
                "confidence_threshold": cfg.confidence_threshold,
                "device": str(device),
                "amp_used": use_amp,
            }
            meta_path = out_dir / "metadata.json"
            meta_path.write_text(json.dumps(metadata, indent=2))
            logger.info("Metadata saved to %s", meta_path)

            # ── Confusion matrix ──────────────────────────────────────────────
            cm_path = Path("model") / "confusion_matrix.png"
            save_confusion_matrix(model, test_loader, device, cm_path)

            # ── MLflow logging (after save — failure here never blocks the model)
            mlflow.log_metrics({
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
            })
            for cat, f1 in test_metrics["per_class_f1"].items():
                mlflow.log_metric(f"test_f1_{_mlflow_key(cat)}", f1)
            mlflow.log_artifact(str(cm_path))
            mlflow.log_artifact(str(meta_path))
            mlflow.log_artifact(str(out_dir), artifact_path="model_artifact")
            mlflow.set_tag("training_status", "SUCCESS")

            # ── Register model to MLflow Model Registry ────────────────────────
            run_id = mlflow.active_run().info.run_id
            try:
                from pos_classifier.model_registry import register_model
                version = register_model(
                    out_dir,
                    run_id,
                    model_name="pos-classifier",
                    description=f"Test accuracy={test_metrics['accuracy']:.4f}, macro_f1={test_metrics['macro_f1']:.4f}",
                )
                mlflow.set_tag("model_version", version.version)
            except Exception:
                logger.exception("Failed to register model to registry (non-fatal).")

        except Exception:
            logger.exception(
                "Training failed at epoch %d — attempting emergency checkpoint save.",
                current_epoch,
            )
            mlflow.set_tag("training_status", "FAILED")
            try:
                _save_last_checkpoint(
                    model, tokenizer, optimizer, scheduler, scaler,
                    current_epoch, best_val_acc, no_improve, last_ckpt_dir,
                )
            except Exception:
                logger.exception("Emergency checkpoint save also failed.")
            raise

    return test_metrics
