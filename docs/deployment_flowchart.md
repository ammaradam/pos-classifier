# Deployment Architecture

This document describes a production deployment design for the POS product description classifier. The local implementation keeps the exercise lightweight with SQLite and local model artifacts, while the production design separates serving, feedback storage, model registry, retraining, monitoring, and promotion gates.

## Kubernetes Deployment Flowchart

```mermaid
flowchart LR
    user[Client applications] --> ingress[Ingress / API Gateway]
    reviewer[Human reviewers] --> review_ui[Review UI / Streamlit]
    ci[CI/CD pipeline] --> registry[Container registry]

    subgraph k8s[Kubernetes cluster]
        subgraph serving[serving namespace]
            ingress --> svc[Service: pos-classifier-api]
            svc --> api[FastAPI Deployment<br/>2-10 replicas via HPA<br/>readiness: /health<br/>metrics: /metrics]
            api --> model_cache[Pod model cache<br/>initContainer downloads approved model]
        end

        subgraph data[data namespace]
            db[(PostgreSQL<br/>predictions, feedback, review queue)]
            object_store[(Object storage<br/>training data and model artifacts)]
            mlflow[(MLflow Tracking<br/>Model Registry)]
        end

        subgraph training[training namespace]
            trigger[Retraining trigger<br/>feedback count 500+ or accuracy below 85%<br/>scheduled weekly CronJob]
            trainer[Kubernetes Job<br/>train + evaluate]
            gate[Promotion gate<br/>quality checks + manual approval]
        end

        subgraph monitoring[monitoring namespace]
            prometheus[Prometheus<br/>scrapes /metrics]
            grafana[Grafana dashboards]
            alertmanager[Alertmanager<br/>routes alerts]
        end
    end

    registry --> api
    registry --> trainer
    api --> db
    api --> prometheus
    api --> review_sample[Review sampler<br/>targets about 20% of predictions]
    review_sample --> db
    review_ui --> db
    db --> trigger
    prometheus --> trigger
    trigger --> trainer
    trainer --> object_store
    trainer --> mlflow
    trainer --> gate
    gate --> mlflow
    gate --> rollout[Rollout approved model version]
    rollout --> api
    prometheus --> grafana
    prometheus --> alertmanager
```

## Monitoring and Feedback Loop

```mermaid
flowchart TD
    request[Prediction request] --> predict[FastAPI /predict or /predict/batch]
    predict --> metrics[Prometheus metrics<br/>request count, latency, validation errors, flag count]
    predict --> persist[Persist prediction event<br/>model version, confidence, category, timestamp]
    persist --> sampler{Send for human review?}
    sampler -->|Low confidence| queue[Review queue]
    sampler -->|Random sample to keep total near 20%| queue
    sampler -->|No| response[Return prediction]
    queue --> review[Human verification]
    review --> feedback[POST /feedback]
    feedback --> feedback_db[(Feedback table)]
    feedback_db --> monitor[Monitoring job/dashboard<br/>accuracy, per-class F1, drift, review SLA]
    metrics --> monitor
    monitor --> retrain{Retrain condition met?}
    retrain -->|feedback count 500+ or accuracy below 85%| job[Training Job]
    retrain -->|weekly schedule| job
    job --> eval[Evaluate candidate model]
    eval --> pass{Pass quality gate?}
    pass -->|Yes| promote[Register and promote model]
    pass -->|No| alert[Alert and keep current model]
    promote --> deploy[Rolling or canary deployment]
    deploy --> predict
```

The review policy should target approximately 20% of total predictions for human verification. Low-confidence predictions get priority; the random sampler fills the remaining review budget so the reviewed set is representative without overwhelming reviewers.

## Production Kubernetes Structure

| Component | Kubernetes resources | Purpose |
|---|---|---|
| API endpoint | `Deployment`, `Service`, `Ingress`, `HPA` | Serves prediction, feedback, health, and metrics endpoints. |
| Model loading | `initContainer`, read-only model cache | Downloads the approved model version before the API starts. |
| Feedback store | PostgreSQL-compatible database | Stores predictions, review queue rows, and verified labels. |
| Training | `CronJob`, on-demand `Job` | Retrains from original data plus verified feedback. |
| Registry | MLflow Tracking and Model Registry | Stores experiments, artifacts, model versions, and stage transitions. |
| Monitoring | Prometheus, Grafana, Alertmanager | Tracks service/model health and routes alerts. |
| Config/security | `Secret`, `ConfigMap`, service account | Manages database credentials, thresholds, registry URI, and cloud access. |

## Model Promotion Strategy

Training jobs must not overwrite the model currently used by serving pods. A safer production flow is:

1. Train a candidate model from approved datasets and human-verified feedback.
2. Log metrics, confusion matrix, and artifacts to MLflow.
3. Evaluate gates such as minimum accuracy, per-class F1, latency, and regression versus the current production model.
4. Register the candidate as a new model version.
5. Promote to `Staging` for validation, then to `Production` after approval.
6. Roll out serving pods with the approved model version using rolling, blue-green, or canary deployment.
7. Keep the previous production version available for rollback.

## Optional GCP Deployment Flowchart

```mermaid
flowchart LR
    clients[Clients] --> lb[Cloud Load Balancing]
    lb --> gke[GKE serving Deployment]
    gke --> sql[(Cloud SQL for PostgreSQL<br/>predictions and feedback)]
    gke --> monitoring[Cloud Monitoring<br/>managed metrics and alerts]
    gke --> logging[Cloud Logging]
    gke --> gcs[(Cloud Storage<br/>datasets and model artifacts)]
    gke --> artifact[Artifact Registry<br/>container images]

    sql --> bq[BigQuery<br/>analytics and reporting]
    bq --> looker[Looker Studio dashboards]

    sql --> pubsub[Pub/Sub feedback events]
    pubsub --> scheduler[Cloud Scheduler / Cloud Run trigger]
    scheduler --> vertex[Vertex AI Pipelines<br/>training and evaluation]
    vertex --> gcs
    vertex --> mlflow[MLflow or Vertex Model Registry]
    mlflow --> approval[Manual approval / promotion]
    approval --> gke
```

## GCP Component Mapping

| Kubernetes/local concept | GCP service | Purpose |
|---|---|---|
| Container image registry | Artifact Registry | Stores immutable API and trainer images. |
| Kubernetes cluster | GKE | Runs FastAPI, review UI, Prometheus-compatible agents, and jobs. |
| Prediction and feedback database | Cloud SQL for PostgreSQL | Durable transactional storage for model events and human labels. |
| Datasets and artifacts | Cloud Storage | Versioned object storage for training data, evaluation outputs, and model files. |
| Training job | Vertex AI Pipelines | Managed retraining, evaluation, lineage, and repeatability. |
| Feedback/retraining events | Pub/Sub | Decouples API traffic from retraining triggers. |
| Metrics and alerts | Cloud Monitoring and Cloud Logging | Centralized SLOs, logs, alerts, and incident routing. |
| Business reporting | BigQuery and Looker Studio | Long-term analytics over prediction and feedback history. |

## Manifest Sketch

The production manifests would be generated with Helm, Kustomize, or GitOps. Key settings:

- API `Deployment`: immutable image tag, readiness/liveness probes on `/health`, Prometheus scrape annotations for `/metrics`, CPU/memory requests and limits, and rolling updates with `maxUnavailable: 0`.
- Model loading: init container downloads the approved `Production` model version from MLflow or object storage into a read-only pod cache.
- Scaling: `HorizontalPodAutoscaler` keeps 2-10 API replicas based on CPU or request latency.
- Training: `CronJob` runs weekly, and an on-demand `Job` can be triggered by feedback volume or accuracy degradation; `concurrencyPolicy: Forbid` prevents overlapping training runs.
- Secrets/config: database credentials, registry URI, sampling rate, confidence threshold, and cloud credentials are provided via `Secret` and `ConfigMap`.
- Release safety: new models are promoted through the registry and rolled out with rolling, blue-green, or canary deployment; serving pods never read from a trainer-written shared PVC.

## Operational Notes

- Keep SQLite only for local development; use PostgreSQL-compatible storage in production.
- Store model artifacts in MLflow/object storage, not on a shared serving PVC.
- Alert on API errors, p95 latency, validation failures, review queue age, model accuracy, class-level regression, and data drift.
- Track review sampling rate so the verified stream remains close to 20% of predictions.
- Require an approval gate before production rollout and keep the previous model available for rollback.
