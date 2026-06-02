# ML Engineering Decisions

This document records the key design decisions made in building the POS product classifier, with the rationale behind each choice. It is intended for reviewers, future maintainers, and anyone extending this system.

---

## 1. Model Selection: `prajjwal1/bert-tiny`

**Decision:** Fine-tune `prajjwal1/bert-tiny` (HuggingFace) for sequence classification with 5 output labels.

**Rationale:**

The task is short-text classification on product descriptions of typically 3–15 words. The label space is small (5 classes) and the training set is large (45K rows), which means the model does not need to be particularly large to perform well.

| Option | Params | Notes |
|--------|--------|-------|
| TF-IDF + Logistic Regression | — | Strong baseline but no contextual understanding; brittle on unseen product names |
| FastText | ~1 MB | Extremely fast; good for keyword-heavy text, but no bidirectional attention |
| **bert-tiny (chosen)** | **4.4M** | Transformer quality; contextual; HuggingFace-downloaded (not TF-embedded); exercise suggestion |
| DistilBERT | 66M | Better accuracy but 15× larger; overkill for 5 clean categories |
| MiniLM-L6 | 22M | Stronger sentence embeddings; still heavier than needed |

BERT-tiny's two transformer layers are sufficient because product descriptions contain strong lexical signals (e.g., "frozen meals", "juice nectars", "dish detergent") that surface within the first attention layer. The bidirectional attention still captures cases where the signal is at the end of a longer description (e.g., "Smart Ones Classic Favorites Mini Rigatoni With Vodka Cream Sauce" → Fresh & Perishable Items).

**Trade-off accepted:** A larger model (e.g., DistilBERT) would likely add 2–3 percentage points of accuracy, but increases Docker image size, inference latency, and memory footprint. Given the narrow label space and strong lexical signals in this dataset, bert-tiny is the right engineering trade-off.

---

## 2. Data Cleaning Strategy

**Decision:** Parse training CSV with `csv.reader` (not `str.split(',')`); apply `unicodedata.normalize('NFKC', ...)` on all text; strip trailing `;;;;` Excel artifacts from category values; drop rows with unrecognised labels silently; deduplicate on exact description text before splitting.

**Rationale:**

The raw CSV has four compounding quality issues:

1. **Trailing semicolons** (`Dry Goods & Pantry Staples;;;;`): Excel export artifact from extra empty columns. A naive parser would treat the category value as `"Dry Goods & Pantry Staples;;;;"`, which won't match `LABEL_MAP` and would silently discard ~40K valid training rows.

2. **Unquoted commas in product descriptions** (e.g., `Cheesecake, Chocolate Truffle,Fresh & Perishable Items;;;;`): Splitting on the first comma would produce `"Cheesecake"` as the description and `"Chocolate Truffle"` as the category. Using `csv.reader` respects quoting rules and handles this correctly.

3. **Non-breaking hyphens** (`‑`, as in `1‑Ply`): These are semantically identical to regular hyphens but won't tokenize the same way. NFKC normalisation maps them to ASCII hyphens, ensuring consistent tokenization.

4. **Duplicate descriptions:** Even when the current dataset has zero exact duplicates, the deduplication step is always run as a defensive invariant. This matters for two reasons: (a) future data refreshes from the same CSV pipeline may reintroduce duplicates; (b) a duplicate that appears in both train and test splits inflates test accuracy by giving the model a memorised example. The implementation keeps the first occurrence; for cross-class duplicates (same text, different labels — a label-noise signal) the first-seen label wins, which is a deterministic tie-break rather than a silent data-quality assumption.

Dropping rows with unknown labels (rather than crashing) is the right default: ~3,000 rows carry a raw category value that is not in `LABEL_MAP` (e.g., brand names like "Organic" or "Buttermilk" that leaked into the category column). Silently skipping them avoids poisoning the training set with wrong labels.

---

## 3. Stratified Train / Val / Test Split

**Decision:** 80% / 10% / 10% split using `sklearn.model_selection.train_test_split` with `stratify=labels`.

**Rationale:**

Without stratification, random sampling can under-represent the minority class (Specialty & Miscellaneous at 5.5%) in the validation or test set, making performance estimates unreliable. With stratification, each split mirrors the original class distribution, so:

- Validation accuracy is a reliable signal for early stopping.
- Test accuracy is a reliable estimate of production performance.
- The model is never evaluated on a set that is easier or harder than what it was trained on.

The 80/10/10 ratio gives 36K training rows (enough for full fine-tuning) and ~4.5K test rows (enough for statistically stable per-class metrics on even the smallest class).

The same stratification principle applies to the `--subset N` smoke-test flag. A naive head-slice of the CSV would under-represent Specialty & Miscellaneous (5.9% of the data) because that class appears later in the file. A head-500 slice observed a 32% relative drop for that class versus the full distribution. The subset is now drawn via `train_test_split(stratify=labels, train_size=N)`, so even quick smoke runs exercise the correct class mix. When stratification is impossible (fewer than 2 samples per class, e.g., in unit-test fixtures), the code falls back to a seeded shuffle with a warning rather than raising.

---

## 4. Class-Weighted Loss Function

**Decision:** Use `sklearn.utils.class_weight.compute_class_weight("balanced", ...)` to compute per-class weights, passed as `weight=` to `torch.nn.CrossEntropyLoss`.

**Rationale:**

The training data is imbalanced:

| Category | Rows | % | Computed weight |
|----------|------|---|-----------------|
| Dry Goods & Pantry Staples | ~14,138 | 31.4% | ~0.79 |
| Fresh & Perishable Items | ~11,041 | 24.5% | ~1.02 |
| Household & Personal Care | ~9,697 | 21.5% | ~1.16 |
| Beverages | ~4,608 | 10.2% | ~2.44 |
| Specialty & Miscellaneous | ~2,483 | 5.5% | ~4.53 |

Without class weights, the cross-entropy gradient is dominated by Dry Goods examples. The model converges to a solution that maximises accuracy on the majority classes while under-learning Specialty & Miscellaneous. Balanced weights scale each class's gradient contribution inversely to its frequency, so the minority class receives proportionally more learning signal per epoch.

**Trade-off accepted:** Class weighting slightly reduces accuracy on majority classes to gain recall on the minority class. This is the correct trade-off for a business application where misclassifying a "Specialty & Miscellaneous" item as "Dry Goods" has real downstream cost (incorrect shelf placement, wrong pricing tier, etc.).

---

## 5. Early Stopping on Validation Accuracy

**Decision:** Stop training when validation accuracy does not improve for 2 consecutive epochs; restore the best checkpoint before saving.

**Rationale:**

Fine-tuning a pre-trained transformer for too many epochs causes overfitting: the model memorises training examples rather than generalising. With 45K training samples and a 4.4M parameter model, overfitting typically begins around epoch 3–5 depending on learning rate.

Early stopping with `patience=2` provides a practical safeguard without requiring a separate learning rate sweep. Restoring the best checkpoint (rather than the final epoch's weights) ensures the saved model corresponds to the lowest validation generalisation error observed during training.

---

## 6. AdamW with Linear Warmup and Decay

**Decision:** AdamW optimiser (`lr=2e-5`, `weight_decay=0.01`) with a linear warmup over 10% of total steps, then linear decay to zero.

**Rationale:**

- **AdamW over Adam:** Weight decay in standard Adam is applied to the adaptive moment estimates rather than the weights themselves, leading to incorrect regularisation. AdamW decouples weight decay from the gradient update, which is the standard approach for transformer fine-tuning.
- **`lr=2e-5`:** This is the empirically established sweet spot for BERT-family fine-tuning on classification tasks. Higher rates (e.g., 1e-4) cause catastrophic forgetting of pre-trained representations; lower rates (e.g., 1e-6) converge too slowly for 5 epochs.
- **Linear warmup (10% of steps):** The pre-trained weights are at a local minimum for their original task. Jumping immediately to the full learning rate applies a large gradient step that can destabilise them. A linear warmup gradually increases the effective learning rate, allowing the model to adapt the new classification head first before adjusting deeper layers.
- **Linear decay to zero:** Gradually reducing the learning rate in later epochs improves final convergence without requiring manual scheduling.

---

## 7. Confidence Thresholding + Random Sampling for Human Review

**Decision:** Flag a prediction for human review if (a) its softmax confidence is below `0.70`, OR (b) it is in a randomly sampled 20% of all predictions.

**Rationale:**

Two independent mechanisms serve different purposes:

| Mechanism | Purpose |
|-----------|---------|
| Confidence threshold (< 0.70) | Route genuinely uncertain predictions. The model signals it cannot reliably distinguish between classes. Always flagged regardless of sample rate. |
| Random 20% sample | Create a representative ground-truth stream. Even high-confidence predictions can be wrong (model overconfidence). The random sample ensures the human-verified set reflects the true production distribution, enabling unbiased accuracy estimation. |

The 0.70 threshold was chosen as a starting point based on the expected calibration of BERT-tiny. It is configurable via `TrainingConfig.confidence_threshold` and should be tuned after the first production deployment by inspecting the calibration curve (confidence vs. empirical accuracy on the verified subset).

The 20% random sample rate matches the exercise specification. In practice, this rate should be informed by the cost of human annotation versus the value of model accuracy improvement.

---

## 8. SQLite for KPI and Feedback Storage

**Decision:** Persist all predictions and human feedback to a local SQLite database (`model/predictions.db`).

**Rationale:**

For an exercise environment (single container, no shared infrastructure), SQLite provides:
- Zero additional infrastructure (no Postgres pod, no connection pooling)
- Atomic writes (WAL mode prevents corruption under concurrent writes from multiple Uvicorn workers)
- Standard SQL queries for the dashboard and retraining trigger logic
- Easy volume-mounting in Docker and Kubernetes

The schema is deliberately simple — two tables (`predictions`, `feedback`) with fixed columns — so that migrating to Postgres for production requires only replacing the `sqlite3` connection string with a `psycopg2` or SQLAlchemy connection. No application logic changes.

**Production upgrade path:** Replace `sqlite3.connect(db_path)` in `predictor.py` and `metrics.py` with a `sqlalchemy.Engine` backed by Cloud SQL or RDS. The rest of the code is unchanged.

---

## 9. Singleton Predictor with Lazy Loading

**Decision:** The `Predictor` class uses a class-level `_instance` variable, loaded once on first call to `Predictor.get()`, and reused for all subsequent requests.

**Rationale:**

Loading a BERT-tiny model from disk takes ~1–2 seconds and consumes ~50MB of memory. Loading it on every request would:
- Add 1–2s latency to every prediction call (unacceptable for real-time serving)
- Waste memory by holding multiple model copies under concurrent load

The singleton pattern ensures a single model instance is shared across all FastAPI request handlers for the lifetime of the process. The `Predictor.reset()` method clears the instance, so that after retraining the next request triggers a fresh load of the new model — no restart required.

---

## 10. Structured Logging with Request Metadata

**Decision:** Use Python's standard `logging` module with a consistent log format (`asctime [level] name — message`). Log every prediction with: description hash, predicted category, confidence score, flagged status, and latency in milliseconds.

**Rationale:**

Unstructured `print()` calls cannot be queried, filtered, or forwarded to a log aggregator (e.g., Cloud Logging, Datadog). Structured logging provides:
- **Observability:** Latency spikes are immediately visible without a profiler.
- **Debugging:** A flagged prediction can be traced end-to-end (request → prediction → feedback) using the description or a request ID.
- **Auditability:** Retraining cycles are logged with timestamps and data row counts.

The description itself is not logged in full (privacy) — only the first 60 characters and the hash are used where needed. In a production system this would be replaced with a proper anonymisation policy.

---

## 11. Feedback-Triggered Retraining

**Decision:** Accumulate human feedback in the `feedback` SQLite table. Trigger retraining when `len(feedback) >= 500` OR when validation accuracy drops below 85%.

**Rationale:**

Retraining too frequently wastes GPU compute and risks overfitting to a small feedback batch. Retraining too rarely allows model drift to accumulate silently. The 500-row threshold provides a practical balance:

- 500 feedback items represents ~10% of the original training data — enough to meaningfully shift the decision boundary.
- The 85% accuracy floor provides a reactive trigger in case a sudden distribution shift causes rapid accuracy degradation.

Both triggers are implemented as checks in `monitoring/metrics.py::pending_retraining_check()`, designed to be polled by a Kubernetes controller or a simple cron job rather than running inside the serving process.

**Human-in-the-loop flywheel:**
```
Model serves predictions
    → 20% sampled for human review
    → Humans correct errors
    → Corrections accumulate in feedback table
    → Retrain on original_data + feedback_data
    → New model improves on previously misclassified patterns
    → Repeat
```

This is intentionally an additive loop: the original training data is always included in retraining to prevent catastrophic forgetting of well-learned patterns from the initial dataset.

---

## 12. Multi-Stage Docker Build

**Decision:** Use a two-stage Dockerfile: a `builder` stage that installs dependencies with `uv`, and a lean `runtime` stage that copies only the venv and source code. Pre-download BERT-tiny weights as a build-time layer.

**Rationale:**

- **Smaller runtime image:** Build tools (`uv`, `pip`, compilers for native extensions) are not needed at runtime. A single-stage build would include them unnecessarily.
- **Layer caching:** Separating dependency installation (`COPY pyproject.toml` → `uv sync`) from source code copying (`COPY src/`) means that rebuilding after a code change does not re-download or re-compile dependencies.
- **Model weights as a layer:** Downloading `prajjwal1/bert-tiny` at build time (∼17MB) bakes the weights into the image. This avoids a network call to HuggingFace Hub on every container start, making cold starts deterministic and eliminating a production dependency on external internet access.

**Trade-off accepted:** The image is larger (∼17MB of model weights included), but this is the correct trade-off for a serving image where startup reliability matters more than image size.

---

## 13. Tokenization Sequence Length: `max_length=64`

**Decision:** Truncate and pad all tokenized inputs to a maximum of 64 tokens.

**Rationale:**

BERT's self-attention is O(n²) in sequence length, so `max_length` has a direct impact on training speed, memory consumption, and inference latency. The default of 128 was reduced to 64 after a data-driven analysis of the training corpus:

| Statistic | Word count | Estimated tokens (×1.4) |
|-----------|-----------|------------------------|
| p50 | 6 | ~8 |
| p90 | 9 | ~13 |
| p95 | 10 | ~14 |
| p99 | 13 | ~18 |
| max | 25 | ~35 |

Even the longest description in the dataset produces fewer than 40 tokens (including `[CLS]` and `[SEP]`). A `max_length` of 64 covers 100% of descriptions with a 1.8× safety margin, while `max_length=128` pads every sequence with 74–120 meaningless `[PAD]` tokens that BERT still processes through its attention layers.

**Effect:** Reducing from 128 to 64 roughly halves the attention computation per forward pass (attention scales as O(n²): (64/128)² = 0.25× the attention matrix operations). On CPU this translates directly to faster training and lower serving latency; on GPU it increases effective batch size under the same memory budget.

**Trade-off accepted:** If the dataset is refreshed with noticeably longer descriptions (e.g., marketing copy rather than product names), this value should be revisited. The correct procedure is to re-run the token-length distribution analysis on the new data and increase `max_length` in `TrainingConfig` if p99 exceeds ~50 tokens.

---

## 14. MLflow Model Registry Integration: Middle-Ground Strategy

**Decision:** Use a dual-mode serving strategy: `docker run` uses the model baked into the image at build time; `docker compose --profile serve` uses the MLflow Model Registry (served by a `mlflow-server` Compose service) and fetches the `Production`-stage model at container startup.

**Rationale:**

Three options were considered:

| Option | Model source | Infra dependency at serve time | Rollout mechanism |
|--------|-------------|-------------------------------|-------------------|
| Baked-only | Image layer | None | Rebuild image |
| Registry-only | MLflow server | MLflow server must be healthy | Push to `Production` stage |
| **Middle-ground (chosen)** | **Registry in Compose; image in docker run** | **Compose only** | **Both paths available** |

The baked-only approach is correct for production deploys (self-contained, no network dependency, deterministic cold start) but makes iterating during local development expensive — every model update requires a full image rebuild and redeploy. The registry-only approach adds a hard runtime dependency: if the MLflow server is unavailable the serving container cannot start, creating an infrastructure coupling that is unacceptable in production.

The middle ground gives each context what it actually needs:

- **`docker run pos-classifier:latest serve`** — ships as a self-contained artefact. The model is in the image layer, startup is deterministic, no MLflow server required. Suitable for production environments, CI smoke tests, and one-shot inference.
- **`docker compose --profile serve up`** — starts an `mlflow-server` container alongside the api. The api reads `MLFLOW_MODEL_NAME=pos-classifier` and `MLFLOW_MODEL_STAGE=Production` from the Compose environment and calls `mlflow.artifacts.download_artifacts("models:/pos-classifier/Production")` at startup. This means a newly trained model can be promoted to `Production` and picked up by a container restart — no image rebuild required.

**Workflow (Compose):**
```
docker compose --profile train up
  → trainer registers best model to mlflow-server
  → trainer auto-promotes to Production

docker compose --profile serve up
  → api fetches Production model from mlflow-server on startup
  → predictions are served immediately
```

**Workflow (docker run / production):**
```
uv run python -m pos_classifier train   # train locally
docker build .                          # bakes model/best_model into image
docker run pos-classifier:latest serve  # no registry, no network dependency
```

**Auto-promotion:** After a successful training run the trainer calls `transition_model_stage("pos-classifier", version, "Production")` immediately. This is intentional for a development/demo stack where every trained model supersedes the previous one. In a production pipeline this step would be gated on a quality threshold (e.g., test macro-F1 > current Production model's macro-F1) before promotion.

**Trade-off accepted:** In Compose, the api has a startup dependency on the MLflow server. If `mlflow-server` is unhealthy the api cannot load the model. This is acceptable in a local/dev context where the full Compose stack is managed together and the operator controls all services. It would not be acceptable in a production deployment, where the baked-image path is used instead.
