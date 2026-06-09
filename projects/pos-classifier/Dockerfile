FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# Build context must be the repo root (see docker-compose.yml build.context).
# All COPY paths are relative to the repo root.

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface \
    MLFLOW_TRACKING_URI=/app/mlruns \
    UV_VERSION=0.11.17 \
    # Install into the container's system Python, not a venv.
    UV_PROJECT_ENVIRONMENT=/usr/local

RUN pip install --no-cache-dir uv==$UV_VERSION

# Copy workspace root manifest + lock + shared library.
# This layer is cached as long as dependencies don't change.
COPY pyproject.toml uv.lock ./
COPY libs/ml-shared/ ./libs/ml-shared/
COPY projects/pos-classifier/pyproject.toml ./projects/pos-classifier/pyproject.toml

RUN uv sync --frozen --package pos-classifier --extra gpu --no-dev

# Copy source code (separate layer — changes more often than deps).
COPY projects/pos-classifier/src/ ./projects/pos-classifier/src/
COPY projects/pos-classifier/model/best_model/ ./model/best_model/

ENV PYTHONPATH=/app/projects/pos-classifier/src

EXPOSE 8000 8501

ENTRYPOINT ["python", "-m", "pos_classifier"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
