"""FastAPI application — prediction serving, feedback, and Prometheus metrics."""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

from pos_classifier.config import LABEL_MAP, TrainingConfig
from pos_classifier.serving.predictor import Predictor
from pos_classifier.serving.schema import (
    BatchPredictRequest,
    BatchPredictResponse,
    FeedbackRequest,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)

logger = logging.getLogger(__name__)

# ── Prometheus metrics ────────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "pos_classifier_requests_total",
    "Total prediction requests",
    ["endpoint", "category"],
)
REQUEST_LATENCY = Histogram(
    "pos_classifier_request_latency_seconds",
    "Prediction request latency",
    ["endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)
FLAGGED_COUNT = Counter(
    "pos_classifier_flagged_total",
    "Predictions flagged for human review",
)
FEEDBACK_COUNT = Counter(
    "pos_classifier_feedback_total",
    "Human feedback submissions received",
)


def create_app(cfg: TrainingConfig | None = None) -> FastAPI:
    if cfg is None:
        cfg = TrainingConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # ── Startup: load model ───────────────────────────────────────────────
        model_dir = cfg.model_output_path()
        db_path = model_dir.parent / "predictions.db"
        try:
            Predictor.get(model_dir=model_dir, db_path=db_path)
            logger.info("Predictor initialised on startup.")
        except Exception as exc:
            logger.warning("Model not found on startup (%s) — load it via retraining.", exc)
        yield  # application runs here
        # ── Shutdown (no-op for now) ──────────────────────────────────────────

    app = FastAPI(
        title="POS Product Classifier",
        description="Fine-tuned BERT-tiny for retail POS product category classification.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request ID middleware ─────────────────────────────────────────────────
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-ms"] = f"{elapsed * 1000:.1f}"
        return response

    # ── Health ────────────────────────────────────────────────────────────────
    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health():
        predictor = Predictor._instance
        return HealthResponse(
            status="ok",
            model_version=predictor.model_version if predictor else "not_loaded",
            model_loaded=predictor is not None,
        )

    # ── Single predict ────────────────────────────────────────────────────────
    @app.post("/predict", response_model=PredictResponse, tags=["prediction"])
    def predict(req: PredictRequest):
        predictor = _get_predictor(cfg)
        t0 = time.perf_counter()
        try:
            results = predictor.predict([req.product_description])
        except Exception as exc:
            logger.exception("Prediction error: %s", exc)
            raise HTTPException(status_code=500, detail="Prediction failed.")
        result = results[0]
        elapsed = time.perf_counter() - t0

        REQUEST_COUNT.labels(endpoint="predict", category=result.category).inc()
        REQUEST_LATENCY.labels(endpoint="predict").observe(elapsed)
        if result.flagged_for_human_review:
            FLAGGED_COUNT.inc()

        logger.info(
            "predict  desc=%r  cat=%s  conf=%.4f  flagged=%s  latency_ms=%.1f",
            req.product_description[:60],
            result.category,
            result.confidence,
            result.flagged_for_human_review,
            elapsed * 1000,
        )
        return result

    # ── Batch predict ─────────────────────────────────────────────────────────
    @app.post("/predict/batch", response_model=BatchPredictResponse, tags=["prediction"])
    def predict_batch(req: BatchPredictRequest):
        predictor = _get_predictor(cfg)
        texts = [item.product_description for item in req.items]
        t0 = time.perf_counter()
        try:
            results = predictor.predict(texts)
        except Exception as exc:
            logger.exception("Batch prediction error: %s", exc)
            raise HTTPException(status_code=500, detail="Batch prediction failed.")
        elapsed = time.perf_counter() - t0

        for r in results:
            REQUEST_COUNT.labels(endpoint="predict_batch", category=r.category).inc()
            if r.flagged_for_human_review:
                FLAGGED_COUNT.inc()
        REQUEST_LATENCY.labels(endpoint="predict_batch").observe(elapsed)

        logger.info(
            "predict_batch  n=%d  latency_ms=%.1f",
            len(texts),
            elapsed * 1000,
        )
        return BatchPredictResponse(results=results, total=len(results))

    # ── Human feedback ────────────────────────────────────────────────────────
    @app.post("/feedback", tags=["feedback"], status_code=204)
    def feedback(req: FeedbackRequest):
        if req.corrected_category not in LABEL_MAP:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown category '{req.corrected_category}'. "
                f"Valid: {list(LABEL_MAP.keys())}",
            )
        predictor = _get_predictor(cfg)
        predictor.record_feedback(
            req.product_description,
            req.corrected_category,
            req.original_prediction,
        )
        FEEDBACK_COUNT.inc()
        logger.info("Feedback received for %r → %s", req.product_description[:60], req.corrected_category)

    # ── Prometheus metrics ────────────────────────────────────────────────────
    @app.get("/metrics", tags=["ops"], include_in_schema=False)
    def metrics():
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


def _get_predictor(cfg: TrainingConfig) -> Predictor:
    model_dir = cfg.model_output_path()
    db_path = model_dir.parent / "predictions.db"
    if Predictor._instance is None:
        if not model_dir.exists():
            raise HTTPException(
                status_code=503,
                detail="Model not trained yet. Run training first.",
            )
        Predictor.get(model_dir=model_dir, db_path=db_path)
    return Predictor._instance  # type: ignore[return-value]
