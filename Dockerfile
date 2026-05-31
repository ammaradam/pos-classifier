FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app/src
ENV HF_HOME=/app/.cache/huggingface
ENV MLFLOW_TRACKING_URI=/app/mlruns
ENV UV_LINK_MODE=copy
ENV UV_COMPILE_BYTECODE=1

RUN pip install --no-cache-dir uv==0.5.4

COPY pyproject.toml uv.lock README.md ./

RUN uv export --frozen --no-dev --no-hashes --no-emit-project -o /tmp/requirements.txt \
    && grep -v '^torch==' /tmp/requirements.txt > /tmp/requirements-no-torch.txt \
    && uv pip install --system --requirements /tmp/requirements-no-torch.txt \
    && uv cache clean \
    && rm /tmp/requirements.txt /tmp/requirements-no-torch.txt

COPY src/ ./src/

RUN uv pip install --system --editable . --no-deps

# Cache the base tokenizer/model needed for retraining. Serving loads the
# fine-tuned model from /app/model, which is created by training or mounted.
RUN python -c "\
from transformers import AutoTokenizer, BertConfig, BertForSequenceClassification; \
from pos_classifier.config import ID_TO_LABEL, LABEL_MAP, NUM_LABELS, TrainingConfig; \
cfg = TrainingConfig(); \
AutoTokenizer.from_pretrained(cfg.tokenizer_name); \
bert_cfg = BertConfig.from_pretrained(cfg.model_name); \
bert_cfg.num_labels = NUM_LABELS; \
bert_cfg.id2label = ID_TO_LABEL; \
bert_cfg.label2id = LABEL_MAP; \
BertForSequenceClassification.from_pretrained(cfg.model_name, config=bert_cfg, ignore_mismatched_sizes=True); \
print('Training model assets cached.')"

RUN mkdir -p /app/data /app/model /app/mlruns

EXPOSE 8000
EXPOSE 8501

ENTRYPOINT ["python", "-m", "pos_classifier"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
