"""
CLI entrypoint for the POS Classifier library.

Usage:
    python -m pos_classifier train   [--data-dir DATA_DIR] [--epochs N] [--subset N]
    python -m pos_classifier serve   [--host HOST] [--port PORT]
    python -m pos_classifier monitor [--port PORT]
    python -m pos_classifier evaluate
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("pos_classifier")


def _cmd_train(args: argparse.Namespace) -> None:
    from pos_classifier.config import TrainingConfig
    from pos_classifier.training.trainer import train

    # Start from env-var defaults; only apply CLI flags that were explicitly set
    # (None sentinel) so that env vars are not silently overridden by argparse defaults.
    overrides: dict = {}
    if args.data_dir is not None:
        overrides["data_dir"] = args.data_dir
    if args.epochs is not None:
        overrides["num_epochs"] = args.epochs
    if args.subset is not None:
        overrides["data_subset"] = args.subset
    if args.model_name is not None:
        overrides["model_name"] = args.model_name
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.lr is not None:
        overrides["learning_rate"] = args.lr

    cfg = TrainingConfig(**overrides)
    logger.info("Starting training with config: %s", cfg)
    metrics = train(cfg)
    logger.info(
        "Training complete. test_accuracy=%.4f  test_macro_f1=%.4f",
        metrics["accuracy"],
        metrics["macro_f1"],
    )


def _cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn
    from pos_classifier.api.app import create_app
    from pos_classifier.config import TrainingConfig

    cfg = TrainingConfig()
    if args.from_registry:
        cfg.mlflow_model_name = args.registry_model
        cfg.mlflow_model_stage = args.registry_stage

    app = create_app(cfg)
    logger.info("Starting API server on http://%s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def _cmd_monitor(args: argparse.Namespace) -> None:
    import subprocess
    import sys
    from pathlib import Path

    dashboard = Path(__file__).parent / "monitoring" / "dashboard.py"
    logger.info("Launching Streamlit dashboard on port %d …", args.port)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(dashboard),
            "--server.port",
            str(args.port),
            "--server.headless",
            "true",
        ],
        check=True,
    )


def _cmd_evaluate(args: argparse.Namespace) -> None:
    """Run inference on the query/validation CSV and print accuracy."""
    from pathlib import Path
    from pos_classifier.config import TrainingConfig
    from pos_classifier.monitoring.metrics import compute_validation_metrics
    from pos_classifier.serving.predictor import Predictor

    cfg = TrainingConfig(data_dir=args.data_dir)
    model_dir = cfg.model_output_path()
    db_path = Path(cfg.db_dir) / "predictions.db"
    query_csv = Path(cfg.data_dir) / "Query_and_Validation_data.csv"

    predictor = Predictor.get(model_dir=model_dir, db_path=db_path)
    metrics = compute_validation_metrics(query_csv, predictor)

    if "error" in metrics:
        logger.error("Evaluation failed: %s", metrics["error"])
        sys.exit(1)

    print(f"\nValidation Results (model version: {metrics['model_version']})")
    print(f"  Overall accuracy : {metrics['overall_accuracy']:.4f}  ({metrics['verified_count']} verified rows)")
    print(f"  Avg confidence   : {metrics['avg_confidence']:.4f}")
    print(f"  Low-conf rate    : {metrics['low_confidence_rate']:.4f}")
    print("\n  Per-class accuracy:")
    for cat, acc in metrics["per_class_accuracy"].items():
        val = f"{acc:.4f}" if acc is not None else "N/A"
        print(f"    {cat:<35} {val}")


def _cmd_register(args: argparse.Namespace) -> None:
    """Register a model from an MLflow run to the Model Registry."""
    from pathlib import Path
    from pos_classifier.config import TrainingConfig
    from pos_classifier.model_registry import register_model, list_model_versions

    cfg = TrainingConfig()
    model_dir = cfg.model_output_path()

    if not model_dir.exists():
        logger.error("Model not found at %s. Run 'train' first.", model_dir)
        sys.exit(1)

    try:
        version = register_model(
            model_dir,
            args.run_id,
            model_name="pos-classifier",
            description=args.description or f"Registered from run {args.run_id}",
        )
        print(f"\n✓ Model registered successfully!")
        print(f"  Name    : pos-classifier")
        print(f"  Version : {version.version}")
        print(f"  Stage   : {version.current_stage}")
        print(f"\nNext: promote to 'Staging' or 'Production' using:")
        print(f"  python -m pos_classifier transition --version {version.version} --stage Staging")

        # Show all versions for reference
        print("\nAll versions:")
        for v in list_model_versions("pos-classifier"):
            print(f"  v{v.version:<3} | {v.current_stage:<10} | {v.creation_timestamp}")
    except Exception as e:
        logger.error("Registration failed: %s", e)
        sys.exit(1)


def _cmd_transition(args: argparse.Namespace) -> None:
    """Transition a model version to a new stage (Staging, Production, Archived)."""
    from pos_classifier.model_registry import transition_model_stage, get_model_version

    if not args.version or not args.stage:
        logger.error("--version and --stage are required")
        sys.exit(1)

    try:
        version_obj = transition_model_stage("pos-classifier", args.version, args.stage)
        print(f"\n✓ Model transitioned successfully!")
        print(f"  Name    : pos-classifier")
        print(f"  Version : {version_obj.version}")
        print(f"  Stage   : {version_obj.current_stage}")
    except ValueError as e:
        logger.error("Transition failed: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.error("Transition failed: %s", e)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pos-classifier",
        description="POS product category classifier CLI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── train ──────────────────────────────────────────────────────────────────
    p_train = sub.add_parser("train", help="Fine-tune the model on training data.")
    p_train.add_argument("--data-dir", default=None, help="Override TRAIN_DATA_DIR env var.")
    p_train.add_argument("--epochs", type=int, default=None, help="Override TRAIN_EPOCHS env var.")
    p_train.add_argument("--subset", type=int, default=None, help="Override TRAIN_SUBSET env var (0 = all).")
    p_train.add_argument("--model-name", default=None, help="Override TRAIN_MODEL_NAME env var.")
    p_train.add_argument("--batch-size", type=int, default=None, help="Override TRAIN_BATCH_SIZE env var.")
    p_train.add_argument("--lr", type=float, default=None, help="Override TRAIN_LR env var.")

    # ── serve ──────────────────────────────────────────────────────────────────
    p_serve = sub.add_parser("serve", help="Start the FastAPI prediction server.")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--from-registry", action="store_true", help="Load model from MLflow registry instead of local path.")
    p_serve.add_argument("--registry-model", default="pos-classifier", help="Registered model name (default: pos-classifier).")
    p_serve.add_argument("--registry-stage", default="Production", choices=["Staging", "Production"], help="Registry stage to load (default: Production).")

    # ── monitor ────────────────────────────────────────────────────────────────
    p_monitor = sub.add_parser("monitor", help="Launch the Streamlit monitoring dashboard.")
    p_monitor.add_argument("--port", type=int, default=8501)

    # ── evaluate ───────────────────────────────────────────────────────────────
    p_eval = sub.add_parser("evaluate", help="Evaluate model on query/validation data.")
    p_eval.add_argument("--data-dir", default="data")

    # ── register ───────────────────────────────────────────────────────────────
    p_register = sub.add_parser("register", help="Manually register a model to the MLflow Model Registry.")
    p_register.add_argument("--run-id", required=True, help="MLflow run ID to register from.")
    p_register.add_argument("--description", help="Optional description of the model version.")

    # ── transition ─────────────────────────────────────────────────────────────
    p_transition = sub.add_parser("transition", help="Transition a model to a new stage (Staging, Production, Archived).")
    p_transition.add_argument("--version", type=int, required=True, help="Model version number.")
    p_transition.add_argument("--stage", required=True, choices=["Staging", "Production", "Archived"], help="Target stage.")

    args = parser.parse_args()

    dispatch = {
        "train": _cmd_train,
        "serve": _cmd_serve,
        "monitor": _cmd_monitor,
        "evaluate": _cmd_evaluate,
        "register": _cmd_register,
        "transition": _cmd_transition,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
