# System Architecture

## Overview

The POS Product Classifier is a text classification library that assigns retail product descriptions to one of five predefined categories. It is structured as an installable Python package with distinct modules for training, serving, and monitoring.

```
┌─────────────────────────────────────────────────────────────────┐
│                      pos-classifier library                      │
│                                                                 │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────────┐  │
│  │    data/    │───▶│  training/   │───▶│     model/        │  │
│  │preprocessing│    │   trainer    │    │  best_model/      │  │
│  │             │    │   evaluate   │    │  metadata.json    │  │
│  └─────────────┘    └──────────────┘    └────────┬──────────┘  │
│                           │                       │             │
│                      MLflow tracking              │             │
│                      (mlruns/)                    ▼             │
│                                          ┌────────────────┐    │
│                                          │   serving/     │    │
│                                          │  predictor     │    │
│                                          │  schema        │    │
│                                          └───────┬────────┘    │
│                                                  │             │
│                                    ┌─────────────┼──────────┐  │
│                                    │             │          │  │
│                               ┌────▼───┐  ┌─────▼────┐     │  │
│                               │  api/  │  │monitoring/│     │  │
│                               │FastAPI │  │ metrics   │     │  │
│                               │/predict│  │ dashboard │     │  │
│                               │/metrics│  │(Streamlit)│     │  │
│                               └────────┘  └───────────┘     │  │
│                                    │             │           │  │
│                               SQLite DB    Prometheus        │  │
│                               (predictions  /metrics         │  │
│                                feedback)                     │  │
│                                                              │  │
└──────────────────────────────────────────────────────────────┘
```

## Module Responsibilities

| Module | File | Responsibility |
|--------|------|---------------|
| Data | `data/preprocessing.py` | CSV parsing, text cleaning, label mapping, `Dataset` class |
| Config | `config.py` | `TrainingConfig` dataclass, `LABEL_MAP`, `ID_TO_LABEL` |
| Trainer | `training/trainer.py` | Fine-tuning loop, MLflow logging, model/tokenizer save |
| Evaluator | `training/evaluate.py` | Accuracy, macro-F1, per-class F1, confusion matrix |
| Predictor | `serving/predictor.py` | Model loading, batch inference, confidence scoring, SQLite KPI storage, feedback recording |
| Schema | `serving/schema.py` | Pydantic request/response models, input validation |
| API | `api/app.py` | FastAPI application, Prometheus counters/histograms |
| Metrics | `monitoring/metrics.py` | Accuracy vs human-verified labels, DB summary, retraining trigger check |
| Dashboard | `monitoring/dashboard.py` | Streamlit UI for real-time monitoring |
| CLI | `__main__.py` | `train \| serve \| monitor \| evaluate` subcommands |

## Data Flow

### Training

```
Training_data.csv
      │
      ▼
preprocessing.py     ← clean CSV, strip artifacts, normalise Unicode
      │
      ├── train (80%) ──▶ fine-tune bert-tiny ──▶ model/best_model/
      ├── val   (10%) ──▶ early stopping
      └── test  (10%) ──▶ final metrics + confusion_matrix.png
                                │
                           mlruns/ (MLflow)
```

### Serving

```
POST /predict
      │
      ▼
schema.py            ← Pydantic validation (min_len=1, max_len=500)
      │
      ▼
predictor.py         ← tokenize ▶ BERT-tiny forward pass ▶ softmax
      │
      ├── confidence < 0.70 → flagged_for_human_review = True
      ├── random 20% sample → flagged_for_human_review = True
      └── all predictions stored to predictions.db (SQLite)
      │
      ▼
PredictResponse      ← {category, confidence, flagged, model_version, predicted_at}
```

### Monitoring & Feedback Loop

```
Query_and_Validation_data.csv (4,688 rows)
              │
              ▼
compute_validation_metrics()
              │
              ├── predict all 4,688 rows
              └── compare vs HUMAN_VERIFIED_Category (~1,166 rows)
                            │
                            ▼
                  overall_accuracy, per_class_accuracy,
                  confidence histogram, prediction_distribution
                            │
                            ▼
                   Streamlit dashboard
                            │
              ┌─────────────┘
              │  POST /feedback (human corrections)
              ▼
     feedback table (SQLite)
              │
       when len(feedback) >= 500  OR  accuracy < 85%
              │
              ▼
     Trigger retraining (Kubernetes Job / manual CLI)
              │
              ▼
     New model/best_model/ + metadata.json
              │
              ▼
     Predictor.reset() → next request loads new model
```

## Technology Stack

| Concern | Technology | Rationale |
|---------|-----------|-----------|
| Model | `prajjwal1/bert-tiny` (HuggingFace) | 4.4M params, transformer quality, lightweight, not TF-embedded |
| ML framework | PyTorch + `transformers` | De-facto standard; full control over training loop |
| Experiment tracking | MLflow (local `mlruns/`) | Open source, self-hosted, no external service needed |
| Serving | FastAPI + Uvicorn | Async, auto-docs, Pydantic native, production standard |
| Schema validation | Pydantic v2 | Type-safe, integrates seamlessly with FastAPI |
| KPI storage | SQLite (`predictions.db`) | Zero infrastructure for the exercise; swap to Postgres for production |
| Monitoring | Prometheus client + Streamlit | Prometheus for time-series metrics; Streamlit for fast dashboard iteration |
| Packaging | `uv` + `hatchling` (src layout) | Fast dependency resolution; clean project structure |
| Containerisation | Docker (multi-stage) | Minimal runtime image; dependencies cached in separate layer |

## Class Imbalance Strategy

The training data is imbalanced (Specialty & Miscellaneous = 5.5% of rows). The trainer computes `sklearn.utils.class_weight.compute_class_weight("balanced", ...)` and passes the result as weights to `torch.nn.CrossEntropyLoss`. This prevents the model from ignoring the minority class.

| Category | Train rows | % | Class weight |
|----------|-----------|---|--------------|
| Dry Goods & Pantry Staples | ~14,138 | 31.4% | ~0.79 |
| Fresh & Perishable Items | ~11,041 | 24.5% | ~1.02 |
| Household & Personal Care | ~9,697 | 21.5% | ~1.16 |
| Beverages | ~4,608 | 10.2% | ~2.44 |
| Specialty & Miscellaneous | ~2,483 | 5.5% | ~4.53 |

## Confidence Thresholding

Every prediction exposes a `confidence` score (max softmax probability). Two complementary mechanisms route predictions to humans:

1. **Low-confidence threshold** (`< 0.70`): Model is genuinely uncertain — always flag.
2. **Random 20% sample**: Even high-confidence predictions get spot-checked, creating a representative human-verified stream for accuracy tracking.

Both types are stored in `predictions.db` with `flagged_for_review = 1` and surface in the Streamlit queue view.
