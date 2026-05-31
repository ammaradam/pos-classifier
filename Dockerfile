# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv==0.4.18

# Copy dependency manifests only (layer caching)
COPY pyproject.toml ./
# Create minimal README to satisfy hatchling
RUN echo "# pos-classifier" > README.md

# Install all dependencies (no editable install yet)
RUN uv sync --no-dev --frozen 2>/dev/null || uv sync --no-dev

# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed venv from builder
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy source code
COPY src/ ./src/
COPY pyproject.toml README.md ./

# Install the package itself (editable equivalent)
RUN pip install --no-cache-dir -e . --no-deps

# Pre-download BERT-tiny weights into the image layer
# (avoids network dependency at runtime inside the container)
RUN python -c "\
from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
AutoTokenizer.from_pretrained('prajjwal1/bert-tiny'); \
AutoModelForSequenceClassification.from_pretrained('prajjwal1/bert-tiny', num_labels=5); \
print('Model weights cached.')"

# Create directories for mounted volumes
RUN mkdir -p /app/data /app/model /app/mlruns

EXPOSE 8000
EXPOSE 8501

# Default command: serve. Override to 'train' or 'monitor'.
ENTRYPOINT ["python", "-m", "pos_classifier"]
CMD ["serve"]
