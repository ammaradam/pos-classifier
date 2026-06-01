# ─────────────────────────────────────────────────────────────────────────────
# Base stage: shared setup (PyTorch + uv)
# ─────────────────────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime AS base

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    HF_HOME=/app/.cache/huggingface \
    MLFLOW_TRACKING_URI=/app/mlruns \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

RUN pip install --no-cache-dir uv==0.5.4

COPY pyproject.toml uv.lock README.md ./

# ─────────────────────────────────────────────────────────────────────────────
# Serve target: API + monitoring (production runtime, no training tools)
# Usage: docker build --target serve -t pos-classifier:serve .
# ─────────────────────────────────────────────────────────────────────────────
FROM base AS serve

# Install only runtime deps (--no-dev excludes training/dev dependencies)
# Skip torch/torchvision — use base image versions
RUN uv export --frozen --no-dev --no-hashes --no-emit-project | \
    grep -v '^torch' | grep -v '^torchvision' | \
    uv pip install --system --requirements /dev/stdin && \
    uv cache clean

COPY src/ ./src/

RUN uv pip install --system --no-deps .

EXPOSE 8000 8501

ENTRYPOINT ["python", "-m", "pos_classifier"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]

# ─────────────────────────────────────────────────────────────────────────────
# Train target: full training environment (includes dev dependencies, metrics libs)
# Usage: docker build --target train -t pos-classifier:train .
# ─────────────────────────────────────────────────────────────────────────────
FROM base AS train

# Install all deps including training/development tools
# Skip torch/torchvision — use base image versions
RUN uv export --frozen --no-hashes --no-emit-project | \
    grep -v '^torch' | grep -v '^torchvision' | \
    uv pip install --system --requirements /dev/stdin && \
    uv cache clean

COPY src/ ./src/

RUN uv pip install --system --no-deps .

EXPOSE 8000

ENTRYPOINT ["python", "-m", "pos_classifier"]
CMD ["train"]
