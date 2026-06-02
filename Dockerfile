FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    HF_HOME=/app/.cache/huggingface \
    MLFLOW_TRACKING_URI=/app/mlruns \
    UV_VERSION=0.11.17

RUN pip install --no-cache-dir uv==$UV_VERSION

COPY pyproject.toml uv.lock README.md ./

RUN uv pip install --system .

COPY src/ ./src/
COPY model/best_model/ ./model/best_model/

EXPOSE 8000 8501

ENTRYPOINT ["python", "-m", "pos_classifier"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
