# POS Product Category Classifier

An end-to-end machine learning service prototype that classifies retail Point-of-Sale (POS) product descriptions into predefined product groups. Built on fine-tuned BERT-tiny, with FastAPI serving, MLflow experiment tracking, Prometheus monitoring, and a Streamlit dashboard.

## Repository Layout

This is a **uv workspace monorepo** with two ML projects and a shared utilities library:

```
├── libs/ml-shared/          shared utilities (config helpers, SQLite retry, MLflow wrappers)
├── projects/pos-classifier/ this project — trained model, API, dashboard, tests
├── projects/brand-detector/ scaffold for a second ML project
├── pyproject.toml           workspace root (no [project] section)
└── uv.lock                  single lock file for all workspace members
```

> **Note:** torch/torchvision are excluded from the workspace lock due to the CPU/GPU dual-index constraint across multiple workspace members. Install torch separately (see Install section) or use Docker (the `pytorch/pytorch` base image provides it).

## Problem

Retail POS data contains free-text product descriptions that must be mapped to a taxonomy of product groups. This service automates classification, logs predictions, sends low-confidence and sampled predictions for human review, and feeds verified labels back into a retraining loop.

**5 Product Categories:**
- Dry Goods & Pantry Staples
- Fresh & Perishable Items
- Household & Personal Care
- Beverages
- Specialty & Miscellaneous

## Architecture

```
Training CSV → BERT-tiny fine-tune → model/best_model/
                                             │
POST /predict ──────────────────────────────▶│
                                        Predictor
                                             │
                       ┌─────────────────────┤
                       │                     │
               predictions.db         PredictResponse
               (SQLite KPI store)      {category, confidence,
                                        flagged_for_human_review}
                       │
               POST /feedback  ←── human annotator
                       │
               when feedback ≥ 500 → retrain trigger
```

See [docs/architecture.md](docs/architecture.md) for the full system diagram and [docs/deployment_flowchart.md](docs/deployment_flowchart.md) for the Kubernetes / GCP deployment design.

## Quickstart

### 1. Install

```bash
# Requires Python 3.11+
pip install uv

# Sync workspace (all non-torch deps)
uv sync

# Install torch separately — CPU:
uv pip install torch torchvision --index https://download.pytorch.org/whl/cpu

# or GPU (CUDA 12.4):
uv pip install torch torchvision==0.21.0 --index https://download.pytorch.org/whl/cu124
```

Expected local data files (relative to `projects/pos-classifier/`):
- `data/Training_data.csv` for model training
- `data/Query_and_Validation_data.csv` for validation, monitoring, and dashboard metrics

### 2. Train

```bash
# Run from the projects/pos-classifier/ directory
cd projects/pos-classifier

python -m pos_classifier train \
  --data-dir data \
  --epochs 5

# Quick smoke-test on 2000 rows:
python -m pos_classifier train --data-dir data --epochs 1 --subset 2000
```

Training logs appear in the console; MLflow tracks all experiments:

```bash
mlflow ui   # → http://localhost:5000
```

### 3. Serve

```bash
cd projects/pos-classifier
python -m pos_classifier serve --port 8000
# → http://localhost:8000/docs  (Swagger UI)
```

**Single prediction:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"product_description": "Chocolate Sandwich Cookies"}'
```

```json
{
  "product_description": "Chocolate Sandwich Cookies",
  "category": "Dry Goods & Pantry Staples",
  "confidence": 0.9841,
  "flagged_for_human_review": false,
  "model_version": "a3f2b1c4",
  "predicted_at": "2024-01-15T10:23:45.123456+00:00"
}
```

**Batch prediction (up to 512 items):**
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"items": [{"product_description": "Sparkling Water"}, {"product_description": "Dish Soap"}]}'
```

**Human feedback submission:**
```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"product_description": "Sparkling Water", "corrected_category": "Beverages"}'
```

### 4. Evaluate

```bash
cd projects/pos-classifier
python -m pos_classifier evaluate --data-dir data
```

### 5. Monitor

```bash
cd projects/pos-classifier
python -m pos_classifier monitor --port 8501
# → http://localhost:8501
```

The Streamlit dashboard shows:
- Overall accuracy vs human-verified labels
- Per-class accuracy breakdown
- Prediction distribution
- Flagged predictions (human review queue)

## Model Versioning & Stage Transitions

Training logs experiment metrics and attempts to register the best model in the **MLflow Model Registry** for versioning. The local API serves the trained artifact under `model/best_model`; the production deployment plan describes how to promote and roll out an approved registry version.

### Stage Workflow

```
Train → Log run + attempt registry registration (v1)
        ↓
Manual: transition v1 to Staging (for validation)
        ↓
Manual: transition v1 to Production (when approved)
```

### Registration during training

When training completes, the best model is saved locally and registration is attempted in the MLflow Model Registry with metrics as the description. Registration failure is non-fatal, so the local model artifact can still be served.

```bash
python -m pos_classifier train --epochs 5
# Logs: "Model registered: pos-classifier v1 (run_id=...)"
```

### Manual registration (if needed)

```bash
# List MLflow runs
mlflow ui  # → http://localhost:5000

# Get the run_id, then:
python -m pos_classifier register --run-id <run_id> --description "Model description"
```

### Promote to Staging / Production

```bash
python -m pos_classifier transition --version 2 --stage Staging
python -m pos_classifier transition --version 2 --stage Production
```

---

## Docker

All Docker commands use the **repo root** as the build context (so the image can access `libs/ml-shared/`). Dockerfiles live inside each project directory.

```bash
# Build the pos-classifier image (from repo root)
docker build -f projects/pos-classifier/Dockerfile -t pos-classifier .

# Train
docker run -v $(pwd)/projects/pos-classifier/data:/app/data \
           -v model_vol:/app/model \
           pos-classifier train --data-dir /app/data

# Serve
docker run -p 8000:8000 pos-classifier serve

# Monitor
docker run -p 8501:8501 \
           -v $(pwd)/projects/pos-classifier/data:/app/data \
           pos-classifier monitor
```

Or use Docker Compose (profiles are project-scoped):

```bash
# Train pos-classifier (exits when done)
docker compose --profile train-pos up pos-trainer

# Serve pos-classifier API + dashboard
docker compose --profile serve-pos up

# Serve both projects simultaneously
docker compose --profile serve-pos --profile serve-bd up
```

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Model health + version |
| `/predict` | POST | Single product classification |
| `/predict/batch` | POST | Batch classification (max 512) |
| `/feedback` | POST | Submit human-verified label |
| `/metrics` | GET | Prometheus metrics |
| `/contract` | GET | OpenAPI schema (JSON) for contract versioning |
| `/docs` | GET | Swagger UI |

### Data Contract & Validation

The API enforces a strict data contract via Pydantic schemas:

**Request validation:**
- `product_description`: 1–500 characters (required)
- Batch size: 1–512 items per request
- All validation errors return `422 Unprocessable Entity` with details

**Response schema:**
- Single prediction: `PredictResponse` with category, confidence (0–1), flagged_for_human_review, model_version, predicted_at
- Batch prediction: `BatchPredictResponse` with results array and total count

**Feedback validation:**
- `corrected_category` must be one of the 5 defined labels (validated at the Predictor layer)
- Invalid categories return `422 Unprocessable Entity` with list of valid options

**Contract export:**
```bash
curl http://localhost:8000/contract > openapi.json
```

### Monitoring & Error Tracking

Prometheus metrics track data quality and validation failures:
- `pos_classifier_validation_errors_total` — schema validation failures by endpoint and error type
- `pos_classifier_requests_total` — total predictions by endpoint and category
- `pos_classifier_flagged_total` — predictions flagged for human review
- `pos_classifier_feedback_total` — human feedback submissions
- `pos_classifier_request_latency_seconds` — prediction latency histogram

Example: Track validation errors over time:
```promql
rate(pos_classifier_validation_errors_total[5m])
```

## Project Structure

```
├── libs/
│   └── ml-shared/               shared utilities (no torch dependency)
│       └── src/ml_shared/
│           ├── config_base.py   env_int / env_float / env_bool helpers
│           ├── db.py            SQLite write-with-retry
│           ├── mlflow_utils.py  register_model / transition_model_stage
│           └── metrics.py      Prometheus factory helpers
│
├── projects/
│   ├── pos-classifier/
│   │   ├── src/pos_classifier/
│   │   │   ├── __main__.py      CLI: train | serve | monitor | evaluate
│   │   │   ├── config.py        TrainingConfig dataclass + label maps
│   │   │   ├── data/
│   │   │   │   └── preprocessing.py  CSV loading, cleaning, tokenization
│   │   │   ├── training/
│   │   │   │   ├── trainer.py   Fine-tuning loop + MLflow logging
│   │   │   │   └── evaluate.py  Metrics: accuracy, F1, confusion matrix
│   │   │   ├── serving/
│   │   │   │   ├── predictor.py Model inference + SQLite KPI storage
│   │   │   │   └── schema.py    Pydantic request/response models
│   │   │   ├── monitoring/
│   │   │   │   ├── metrics.py   Accuracy vs human labels, DB summary
│   │   │   │   └── dashboard.py Streamlit monitoring dashboard
│   │   │   └── api/
│   │   │       └── app.py       FastAPI application
│   │   ├── tests/
│   │   ├── data/
│   │   ├── model/best_model/
│   │   ├── notebooks/eda.ipynb
│   │   └── Dockerfile
│   │
│   └── brand-detector/          second project scaffold
│       ├── src/brand_detector/
│       └── Dockerfile
│
├── pyproject.toml               workspace root
├── uv.lock
└── docker-compose.yml
```

## Tests

```bash
# Run from repo root
uv run pytest projects/pos-classifier -v

# Or from inside the project
cd projects/pos-classifier
uv run pytest tests/ -v
```

## Model Details

| Property | Value |
|----------|-------|
| Base model | `prajjwal1/bert-tiny` |
| Parameters | 4.4 million |
| Max sequence length | 64 tokens |
| Training data | ~42,000 rows (after cleaning) |
| Split | 80% train / 10% val / 10% test |
| Optimizer | AdamW, lr=2e-5, weight_decay=0.01 |
| Loss | CrossEntropyLoss with class weights |
| Early stopping | Patience=2 on val accuracy |
| Expected test accuracy | > 90% |

## Configuration

All training hyperparameters are exposed via `TrainingConfig` in [projects/pos-classifier/src/pos_classifier/config.py](projects/pos-classifier/src/pos_classifier/config.py) and as CLI flags:

```bash
python -m pos_classifier train --help
```

Key environment variables:
- `MLFLOW_TRACKING_URI` — MLflow server URI (default: `mlruns/`)
- `TRAIN_DATA_DIR` — path to training data directory
- `DB_DIR` — path for SQLite predictions database (default: `db/`)

## Deployment

See [docs/deployment_flowchart.md](docs/deployment_flowchart.md) for:
- Kubernetes deployment with HPA, CronJob training, model registry promotion, and Prometheus/Grafana monitoring
- Monitoring and feedback flow with about 20% human verification and retraining triggers
- Optional GCP architecture with GKE, Cloud SQL, Cloud Storage, Vertex AI Pipelines, Pub/Sub, and BigQuery
