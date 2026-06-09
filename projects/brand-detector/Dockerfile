FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# Build context must be the repo root (see docker-compose.yml build.context).

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface \
    MLFLOW_TRACKING_URI=/app/mlruns \
    UV_VERSION=0.11.17 \
    UV_PROJECT_ENVIRONMENT=/usr/local

RUN pip install --no-cache-dir uv==$UV_VERSION

COPY pyproject.toml uv.lock ./
COPY libs/ml-shared/ ./libs/ml-shared/
COPY projects/brand-detector/pyproject.toml ./projects/brand-detector/pyproject.toml

RUN uv sync --frozen --package brand-detector --extra gpu --no-dev

COPY projects/brand-detector/src/ ./projects/brand-detector/src/

ENV PYTHONPATH=/app/projects/brand-detector/src

EXPOSE 8001

ENTRYPOINT ["python", "-m", "brand_detector"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8001"]
