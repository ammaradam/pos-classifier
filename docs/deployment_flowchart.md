# Deployment Architecture

## Kubernetes Deployment

```
                        ┌────────────────────────────────────────────────────────────────┐
                        │                    Kubernetes Cluster                           │
                        │                                                                │
   ┌──────────┐         │  ┌─────────────────────────────────────────────────────────┐  │
   │   User   │─────────┼─▶│                   Ingress (NGINX)                       │  │
   │  / CI    │         │  │  api.example.com  /predict  /predict/batch  /feedback   │  │
   └──────────┘         │  └──────────────────────┬──────────────────────────────────┘  │
                        │                         │                                      │
                        │  ┌──────────────────────▼──────────────────────────────────┐  │
                        │  │              serving namespace                           │  │
                        │  │                                                         │  │
                        │  │  ┌─────────────────────────────────────────────┐        │  │
                        │  │  │   FastAPI Deployment  (HPA: 2→10 replicas)  │        │  │
                        │  │  │   Image: pos-classifier:latest              │        │  │
                        │  │  │   CMD: serve                                │        │  │
                        │  │  │   ReadinessProbe: GET /health               │        │  │
                        │  │  │   Resources: 500m CPU / 512Mi RAM per pod   │        │  │
                        │  │  └───────────────────────┬─────────────────────┘        │  │
                        │  │                          │                              │  │
                        │  │  ┌───────────────────────▼─────────────────────┐        │  │
                        │  │  │           PersistentVolumeClaim              │        │  │
                        │  │  │  /app/model  (model artifacts + metadata)   │        │  │
                        │  │  │  /app/model/predictions.db  (KPI + feedback)│        │  │
                        │  │  └─────────────────────────────────────────────┘        │  │
                        │  └──────────────────────────────────────────────────────── ┘  │
                        │                                                                │
                        │  ┌─────────────────────────────────────────────────────────┐  │
                        │  │              training namespace                          │  │
                        │  │                                                         │  │
                        │  │  Trigger conditions (either):                           │  │
                        │  │    A) feedback rows ≥ 500  →  Kubernetes Job            │  │
                        │  │    B) accuracy < 85%       →  Kubernetes Job            │  │
                        │  │    C) Scheduled CronJob    →  weekly retraining         │  │
                        │  │                                                         │  │
                        │  │  ┌──────────────────────────────────────────────┐       │  │
                        │  │  │   Training Job / CronJob                     │       │  │
                        │  │  │   Image: pos-classifier:latest               │       │  │
                        │  │  │   CMD: train --data-dir /app/data --epochs 5 │       │  │
                        │  │  │   backoffLimit: 2                            │       │  │
                        │  │  └───────────────┬──────────────────────────────┘       │  │
                        │  │                  │ writes new model to PVC              │  │
                        │  │                  ▼                                      │  │
                        │  │  ┌──────────────────────────────────────────────┐       │  │
                        │  │  │   Shared PVC: /app/model, /app/data (RO)    │       │  │
                        │  │  └──────────────────────────────────────────────┘       │  │
                        │  └──────────────────────────────────────────────────────── ┘  │
                        │                                                                │
                        │  ┌─────────────────────────────────────────────────────────┐  │
                        │  │              monitoring namespace                        │  │
                        │  │                                                         │  │
                        │  │  ┌─────────────────────┐  ┌──────────────────────────┐  │  │
                        │  │  │    Prometheus Pod    │  │   Streamlit Dashboard    │  │  │
                        │  │  │  scrape /metrics     │  │   CMD: monitor           │  │  │
                        │  │  │  every 15s           │  │   monitor.example.com    │  │  │
                        │  │  │  alert rules:        │  │                          │  │  │
                        │  │  │  • accuracy < 85%    │  │   Shows:                 │  │  │
                        │  │  │  • error_rate > 1%   │  │   • accuracy vs baseline │  │  │
                        │  │  │  • latency p95 > 1s  │  │   • per-class F1         │  │  │
                        │  │  └─────────────────────┘  │   • confidence histogram  │  │  │
                        │  │                           │   • flagged queue         │  │  │
                        │  │                           └──────────────────────────┘  │  │
                        │  └──────────────────────────────────────────────────────── ┘  │
                        └────────────────────────────────────────────────────────────────┘
```

## Monitoring & Feedback Loop (detailed)

```
Prediction Request
        │
        ▼
   FastAPI /predict
        │
        ├─▶ Prometheus counter: pos_classifier_requests_total{category}
        ├─▶ Prometheus histogram: pos_classifier_request_latency_seconds
        │
        ▼
   SQLite predictions.db
        │
        ├── flagged_for_review = 1  (low-confidence OR random 20% sample)
        │
        ▼
   Human Annotator
   views Streamlit queue
        │
        ▼
   POST /feedback  {corrected_category}
        │
        ▼
   SQLite feedback table
        │
   when len >= 500
        │
        ▼
   Kubernetes Job triggered
   (training namespace)
        │
        ▼
   train with original_data + feedback_data
        │
        ▼
   New model/best_model/ + metadata.json written to PVC
        │
        ▼
   Predictor.reset() called
   Next prediction request loads new model
```

## Optional GCP Deployment

```
┌───────────────────────────────────────────────────────────────────┐
│                          GCP Architecture                          │
│                                                                   │
│  ┌─────────────┐     ┌──────────────┐     ┌───────────────────┐  │
│  │  Cloud      │     │    GKE       │     │   Cloud Storage   │  │
│  │  Load       │────▶│  Cluster     │────▶│   (GCS Bucket)    │  │
│  │  Balancer   │     │  (serving    │     │   model artifacts │  │
│  └─────────────┘     │   pods)      │     │   training data   │  │
│                      └──────┬───────┘     └───────────────────┘  │
│                             │                                     │
│                      ┌──────▼───────┐     ┌───────────────────┐  │
│                      │  Vertex AI   │     │    Cloud SQL      │  │
│                      │  Pipelines   │     │   (PostgreSQL)    │  │
│                      │  (training   │     │  predictions +    │  │
│                      │   pipeline)  │     │  feedback tables  │  │
│                      └──────────────┘     └───────────────────┘  │
│                                                                   │
│  ┌─────────────┐     ┌──────────────┐     ┌───────────────────┐  │
│  │  Pub/Sub    │     │  BigQuery    │     │  Cloud Monitoring │  │
│  │  (feedback  │────▶│  (long-term  │     │  + Looker Studio  │  │
│  │   events)   │     │  analytics)  │     │  (dashboards)     │  │
│  └─────────────┘     └──────────────┘     └───────────────────┘  │
└───────────────────────────────────────────────────────────────────┘
```

### GCP Component Mapping

| Local | GCP equivalent | Purpose |
|-------|---------------|---------|
| PVC (model artifacts) | Cloud Storage (GCS) | Durable, versioned model storage |
| SQLite predictions.db | Cloud SQL (PostgreSQL) | Scalable relational storage |
| Prometheus | Cloud Monitoring | Managed metrics, no infra |
| Streamlit dashboard | Looker Studio + BigQuery | Enterprise BI, shareable |
| Kubernetes Job | Vertex AI Pipelines | Managed ML pipeline with lineage |
| Random feedback sampling | Pub/Sub → Cloud Function | Event-driven, decoupled |
| Weekly CronJob | Cloud Scheduler → Cloud Run | Serverless trigger |

## Kubernetes YAML Sketches

### Serving Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pos-classifier-api
  namespace: serving
spec:
  replicas: 2
  selector:
    matchLabels:
      app: pos-classifier-api
  template:
    spec:
      containers:
        - name: api
          image: pos-classifier:latest
          command: ["python", "-m", "pos_classifier", "serve"]
          ports:
            - containerPort: 8000
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
          resources:
            requests:
              cpu: "500m"
              memory: "512Mi"
            limits:
              cpu: "1"
              memory: "1Gi"
          volumeMounts:
            - name: model-vol
              mountPath: /app/model
      volumes:
        - name: model-vol
          persistentVolumeClaim:
            claimName: model-pvc
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: pos-classifier-hpa
  namespace: serving
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: pos-classifier-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

### Training CronJob

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: pos-classifier-weekly-train
  namespace: training
spec:
  schedule: "0 2 * * 0"   # Sundays at 02:00 UTC
  jobTemplate:
    spec:
      backoffLimit: 2
      template:
        spec:
          restartPolicy: OnFailure
          containers:
            - name: trainer
              image: pos-classifier:latest
              command: ["python", "-m", "pos_classifier", "train",
                        "--data-dir", "/app/data", "--epochs", "5"]
              volumeMounts:
                - name: model-vol
                  mountPath: /app/model
                - name: data-vol
                  mountPath: /app/data
                  readOnly: true
          volumes:
            - name: model-vol
              persistentVolumeClaim:
                claimName: model-pvc
            - name: data-vol
              persistentVolumeClaim:
                claimName: data-pvc
```
