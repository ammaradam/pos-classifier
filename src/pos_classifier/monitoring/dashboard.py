"""
Streamlit monitoring dashboard.

Run with:
    streamlit run src/pos_classifier/monitoring/dashboard.py
"""

import logging
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Suppress transformers' __path__ lazy-loading warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

from pos_classifier.config import ID_TO_LABEL, TrainingConfig
from pos_classifier.monitoring.metrics import compute_validation_metrics, get_db_summary
from pos_classifier.serving.predictor import Predictor

st.set_page_config(
    page_title="POS Classifier — Monitor",
    page_icon="📊",
    layout="wide",
)

st.title("📊 POS Product Classifier — Monitoring Dashboard")

cfg = TrainingConfig()
model_dir = cfg.model_output_path()
db_path = model_dir.parent / "predictions.db"
query_csv = Path(cfg.data_dir) / "Query_and_Validation_data.csv"

# ── Sidebar: controls ─────────────────────────────────────────────────────────
st.sidebar.header("Controls")
run_validation = st.sidebar.button("🔄 Run Validation on Query Data", type="primary")
st.sidebar.markdown("---")
st.sidebar.caption("Validation compares model predictions against human-verified labels (≈1,166 rows).")

# ── Model status ──────────────────────────────────────────────────────────────
if not model_dir.exists():
    st.error("⚠️  No trained model found. Run `python -m pos_classifier train` first.")
    st.stop()

predictor = Predictor.get(model_dir=model_dir, db_path=db_path)
st.success(f"✅ Model loaded — version **{predictor.model_version}**  |  confidence threshold: **{predictor.confidence_threshold:.0%}**")

# ── DB summary ────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📦 Prediction Database Summary")
summary = get_db_summary(db_path)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Predictions", summary["total_predictions"])
col2.metric("Flagged for Review", summary["flagged_count"])
col3.metric("Human Feedback Items", summary["total_feedback"])
col4.metric(
    "Flag Rate",
    f"{summary['flagged_count'] / max(summary['total_predictions'], 1):.1%}",
)

if summary.get("category_distribution"):
    dist = summary["category_distribution"]
    fig_dist = px.bar(
        x=list(dist.keys()),
        y=list(dist.values()),
        labels={"x": "Category", "y": "Count"},
        title="Prediction Distribution (all DB rows)",
        color=list(dist.values()),
        color_continuous_scale="Blues",
    )
    fig_dist.update_layout(showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig_dist, width='stretch')

# ── Validation results ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("✅ Validation Accuracy (vs Human-Verified Labels)")

if run_validation:
    if not query_csv.exists():
        st.error(f"Query CSV not found at {query_csv}")
    else:
        with st.spinner("Running inference on query data …"):
            metrics = compute_validation_metrics(query_csv, predictor)
        st.session_state["val_metrics"] = metrics

val_metrics = st.session_state.get("val_metrics")
if val_metrics is None:
    st.info("Click **Run Validation** to compute accuracy against the human-verified subset.")
elif "error" in val_metrics:
    st.error(val_metrics["error"])
else:
    acc = val_metrics["overall_accuracy"]
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Overall Accuracy", f"{acc:.1%}", delta=None)
    col_b.metric("Verified Rows", val_metrics["verified_count"])
    col_c.metric("Avg Confidence", f"{val_metrics['avg_confidence']:.2%}")
    col_d.metric("Low-Confidence Rate", f"{val_metrics['low_confidence_rate']:.1%}")

    # Per-class accuracy bar
    per_cls = {k: v for k, v in val_metrics["per_class_accuracy"].items() if v is not None}
    fig_cls = px.bar(
        x=list(per_cls.keys()),
        y=list(per_cls.values()),
        title="Per-Class Accuracy on Verified Subset",
        labels={"x": "Category", "y": "Accuracy"},
        color=list(per_cls.values()),
        color_continuous_scale="RdYlGn",
        range_color=[0.5, 1.0],
    )
    fig_cls.add_hline(y=0.85, line_dash="dash", line_color="red", annotation_text="85% threshold")
    fig_cls.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_cls, width='stretch')

    # Prediction distribution on full query set
    st.subheader("📈 Prediction Distribution on Full Query Set")
    qdist = val_metrics["prediction_distribution"]
    fig_q = px.pie(
        names=list(qdist.keys()),
        values=list(qdist.values()),
        title="Query Set — Predicted Category Distribution",
        hole=0.4,
    )
    st.plotly_chart(fig_q, width='stretch')

# ── Feedback queue ────────────────────────────────────────────────────────────
import sqlite3
st.markdown("---")
st.subheader("🔔 Flagged Predictions (Human Review Queue)")
if db_path.exists():
    conn = sqlite3.connect(db_path)
    flagged_rows = conn.execute(
        """SELECT product_description, predicted_category, confidence, predicted_at
           FROM predictions WHERE flagged_for_review = 1
           ORDER BY predicted_at DESC LIMIT 100"""
    ).fetchall()
    conn.close()

    if flagged_rows:
        import pandas as pd
        df = pd.DataFrame(
            flagged_rows,
            columns=["Product Description", "Predicted Category", "Confidence", "Predicted At"],
        )
        df["Confidence"] = df["Confidence"].map("{:.2%}".format)
        st.dataframe(df, width='stretch')
    else:
        st.info("No flagged predictions yet.")
else:
    st.info("No prediction database found. Start the API and make some predictions.")

st.markdown("---")
st.caption("Refresh the page or click 'Run Validation' to update metrics.")
