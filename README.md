# POS Product Category Classifier

A production-ready machine learning library that classifies retail Point-of-Sale (POS) product descriptions into predefined product groups. Built on fine-tuned BERT-tiny, with FastAPI serving, MLflow experiment tracking, Prometheus monitoring, and a Streamlit dashboard.

## Problem

Retail POS data contains free-text product descriptions that must be mapped to a taxonomy of product groups. This library automates that classification (∼80% of cases) and routes uncertain predictions to human reviewers (∼20%), feeding a continuous-improvement loop.

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

See [doc/architecture.md](doc/architecture.md) for the full system diagram and [doc/deployment_flowchart.md](doc/deployment_flowchart.md) for the Kubernetes / GCP deployment design.

## Quickstart

### 1. Install

```bash
# Requires Python 3.11+
pip install uv
uv sync
uv pip install -e .
```

### 2. Train

```bash
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

Compute accuracy against the human-verified subset of the query data:

```bash
python -m pos_classifier evaluate --data-dir data
```

### 5. Monitor

```bash
python -m pos_classifier monitor --port 8501
# → http://localhost:8501
```

The Streamlit dashboard shows:
- Overall accuracy vs human-verified labels
- Per-class accuracy breakdown
- Prediction distribution
- Flagged predictions (human review queue)

## Model Versioning & Stage Transitions

All trained models are automatically registered to the **MLflow Model Registry** for production deployment and versioning.

### Stage Workflow

```
Train → Auto-register (v1)
        ↓
Manual: transition v1 to Staging (for validation)
        ↓
Manual: transition v1 to Production (when approved)
```

### Auto-registration on training

When training completes, the best model is automatically registered to the MLflow Model Registry with metrics as the description:

```bash
python -m pos_classifier train --epochs 5
# Logs: "Model registered: pos-classifier v1 (run_id=...)"
```

### Manual registration (if needed)

If you need to register a model from an existing run without retraining:

```bash
# List MLflow runs
mlflow ui  # → http://localhost:5000

# Get the run_id, then:
python -m pos_classifier register --run-id <run_id> --description "Model description"

# Output:
# ✓ Model registered successfully!
#   Name    : pos-classifier
#   Version : 2
#   Stage   : None
#
# All versions:
#   v1  | Production | 1737461234000
#   v2  | None       | 1737461567000
```

### Promote to Staging

After registering, validate the model on a subset of query data, then promote to **Staging**:

```bash
python -m pos_classifier evaluate --data-dir data
# Check the per-class accuracy metrics

# If metrics look good:
python -m pos_classifier transition --version 2 --stage Staging

# Output:
# ✓ Model transitioned successfully!
#   Name    : pos-classifier
#   Version : 2
#   Stage   : Staging
```

### Promote to Production

Once validated in Staging, promote to **Production**:

```bash
python -m pos_classifier transition --version 2 --stage Production

# ✓ Model transitioned successfully!
#   Name    : pos-classifier
#   Version : 2
#   Stage   : Production
```

The API automatically loads the **Production** stage model on startup. If you serve a specific version:

```python
# In serving/predictor.py, the Predictor uses Production stage by default
# To serve a Staging model instead, manually specify the version URI in MLflow
```

### View all model versions

```bash
mlflow ui  # → http://localhost:5000
# Browse Models → pos-classifier → see all versions, stages, metrics
```

---

## Docker

```bash
# Build once
docker build -t pos-classifier .

# Train
docker run -v $(pwd)/data:/app/data -v model_vol:/app/model \
  pos-classifier train --data-dir /app/data

# Serve
docker run -p 8000:8000 -v model_vol:/app/model pos-classifier serve

# Monitor
docker run -p 8501:8501 -v $(pwd)/data:/app/data -v model_vol:/app/model \
  pos-classifier monitor
```

Or use Docker Compose:

```bash
# Train (exits when done)
docker compose --profile train up trainer

# Serve API + dashboard
docker compose --profile serve up
```

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Model health + version |
| `/predict` | POST | Single product classification |
| `/predict/batch` | POST | Batch classification (max 512) |
| `/feedback` | POST | Submit human-verified label |
| `/metrics` | GET | Prometheus metrics |
| `/docs` | GET | Swagger UI |

## Project Structure

```
src/pos_classifier/
├── __init__.py
├── __main__.py          ← CLI: train | serve | monitor | evaluate
├── config.py            ← TrainingConfig dataclass + label maps
├── data/
│   └── preprocessing.py ← CSV loading, cleaning, tokenization
├── training/
│   ├── trainer.py       ← Fine-tuning loop + MLflow logging
│   └── evaluate.py      ← Metrics: accuracy, F1, confusion matrix
├── serving/
│   ├── predictor.py     ← Model inference + SQLite KPI storage
│   └── schema.py        ← Pydantic request/response models
├── monitoring/
│   ├── metrics.py       ← Accuracy vs human labels, DB summary
│   └── dashboard.py     ← Streamlit monitoring dashboard
└── api/
    └── app.py           ← FastAPI application

tests/
├── test_preprocessing.py
├── test_schema.py
└── test_api.py

doc/
├── architecture.md          ← System design + data flow diagrams
└── deployment_flowchart.md  ← Kubernetes + GCP deployment design
```

## Tests

```bash
uv run pytest tests/ -v
```

## Model Details

| Property | Value |
|----------|-------|
| Base model | `prajjwal1/bert-tiny` |
| Parameters | 4.4 million |
| Max sequence length | 128 tokens |
| Training data | ~42,000 rows (after cleaning) |
| Split | 80% train / 10% val / 10% test |
| Optimizer | AdamW, lr=2e-5, weight_decay=0.01 |
| Loss | CrossEntropyLoss with class weights |
| Early stopping | Patience=2 on val accuracy |
| Expected test accuracy | > 90% |

## Configuration

All training hyperparameters are exposed via `TrainingConfig` in [config.py](src/pos_classifier/config.py) and as CLI flags:

```bash
python -m pos_classifier train --help
```

Key environment variables:
- `MLFLOW_TRACKING_URI` — MLflow server URI (default: `mlruns/`)

## Deployment

See [doc/deployment_flowchart.md](doc/deployment_flowchart.md) for:
- Kubernetes deployment with HPA, CronJob training, Prometheus monitoring
- Optional GCP architecture with GKE, Cloud Storage, Vertex AI Pipelines, BigQuery
