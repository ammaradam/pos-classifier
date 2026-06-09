"""
CLI entrypoint — run as 'pos-classifier <command>' after installing the package.

Commands:
    train      Fine-tune BERT-tiny on the product-description dataset
    serve      Start the FastAPI prediction server
    evaluate   Score the model on the validation CSV
    monitor    Launch the Streamlit monitoring dashboard
    register   Register a training run to the MLflow Model Registry
    transition Promote / archive a registered model version

Run 'pos-classifier <command> --help' for per-command options and examples.
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
        description=(
            "POS product category classifier — fine-tuned BERT-tiny with FastAPI serving and MLflow tracking.\n"
            "\n"
            "Typical workflow:\n"
            "  1. pos-classifier train                 fine-tune the model\n"
            "  2. pos-classifier serve                 start the prediction API\n"
            "  3. pos-classifier evaluate              score the validation set\n"
            "  4. pos-classifier monitor               open the live dashboard\n"
            "  5. pos-classifier register --run-id ID  publish model to the registry\n"
            "  6. pos-classifier transition --version N --stage Production\n"
        ),
        epilog="Run 'pos-classifier <command> --help' for detailed options on any command.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── train ──────────────────────────────────────────────────────────────────
    p_train = sub.add_parser(
        "train",
        help="Fine-tune the model on training data.",
        description=(
            "Fine-tune BERT-tiny on the POS product-description dataset. "
            "All flags are optional — unset flags fall back to environment variables, "
            "which in turn fall back to the defaults in TrainingConfig. "
            "The run is tracked in MLflow; the best checkpoint is saved to the model output directory."
        ),
        epilog=(
            "Examples:\n"
            "  pos-classifier train\n"
            "  pos-classifier train --data-dir data/raw --epochs 5\n"
            "  pos-classifier train --subset 1000 --batch-size 32 --lr 3e-5\n"
            "\n"
            "Environment variables (checked when the flag is omitted):\n"
            "  TRAIN_DATA_DIR   path to the directory containing the CSV files\n"
            "  TRAIN_EPOCHS     number of training epochs\n"
            "  TRAIN_SUBSET     rows to sample per split (0 = use all data)\n"
            "  TRAIN_MODEL_NAME HuggingFace model ID to fine-tune\n"
            "  TRAIN_BATCH_SIZE mini-batch size\n"
            "  TRAIN_LR         peak learning rate"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_train.add_argument(
        "--data-dir", default=None, metavar="PATH",
        help="Directory containing training CSV files. Overrides TRAIN_DATA_DIR.",
    )
    p_train.add_argument(
        "--epochs", type=int, default=None, metavar="N",
        help="Number of full passes over the training data. Overrides TRAIN_EPOCHS.",
    )
    p_train.add_argument(
        "--subset", type=int, default=None, metavar="N",
        help="Randomly sample N rows per split for a quick smoke-test (0 = all rows). Overrides TRAIN_SUBSET.",
    )
    p_train.add_argument(
        "--model-name", default=None, metavar="MODEL_ID",
        help="HuggingFace model ID to fine-tune (e.g. 'prajjwal1/bert-tiny'). Overrides TRAIN_MODEL_NAME.",
    )
    p_train.add_argument(
        "--batch-size", type=int, default=None, metavar="N",
        help="Mini-batch size for training and evaluation. Overrides TRAIN_BATCH_SIZE.",
    )
    p_train.add_argument(
        "--lr", type=float, default=None, metavar="LR",
        help="Peak learning rate for the AdamW optimiser (e.g. 3e-5). Overrides TRAIN_LR.",
    )

    # ── serve ──────────────────────────────────────────────────────────────────
    p_serve = sub.add_parser(
        "serve",
        help="Start the FastAPI prediction server.",
        description=(
            "Launch the FastAPI REST API that exposes the trained classifier. "
            "By default the model is loaded from the local checkpoint directory. "
            "Pass --from-registry to instead pull a version from the MLflow Model Registry."
        ),
        epilog=(
            "Examples:\n"
            "  pos-classifier serve\n"
            "  pos-classifier serve --port 9000\n"
            "  pos-classifier serve --from-registry --registry-stage Staging\n"
            "\n"
            "Key endpoints (available at http://<host>:<port>):\n"
            "  POST /predict          classify a single product description\n"
            "  POST /predict/batch    classify a list of descriptions in one call\n"
            "  GET  /health           liveness check\n"
            "  GET  /metrics          Prometheus metrics scrape endpoint\n"
            "  GET  /docs             interactive Swagger UI"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_serve.add_argument(
        "--host", default="0.0.0.0", metavar="HOST",
        help="Interface to bind to (default: 0.0.0.0 — all interfaces).",
    )
    p_serve.add_argument(
        "--port", type=int, default=8000, metavar="PORT",
        help="TCP port to listen on (default: 8000).",
    )
    p_serve.add_argument(
        "--from-registry", action="store_true",
        help="Load the model from the MLflow Model Registry instead of the local checkpoint path.",
    )
    p_serve.add_argument(
        "--registry-model", default="pos-classifier", metavar="NAME",
        help="Registered model name in MLflow (default: pos-classifier).",
    )
    p_serve.add_argument(
        "--registry-stage", default="Production", choices=["Staging", "Production"],
        help="Registry stage to load — 'Staging' or 'Production' (default: Production).",
    )

    # ── monitor ────────────────────────────────────────────────────────────────
    p_monitor = sub.add_parser(
        "monitor",
        help="Launch the Streamlit monitoring dashboard.",
        description=(
            "Open a Streamlit dashboard that visualises live prediction traffic, "
            "confidence distributions, per-class accuracy trends, and data-drift indicators "
            "sourced from the local SQLite predictions database."
        ),
        epilog=(
            "Examples:\n"
            "  pos-classifier monitor\n"
            "  pos-classifier monitor --port 8502\n"
            "\n"
            "The API server must be running (pos-classifier serve) to populate the database\n"
            "that the dashboard reads from."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_monitor.add_argument(
        "--port", type=int, default=8501, metavar="PORT",
        help="Port for the Streamlit server (default: 8501).",
    )

    # ── evaluate ───────────────────────────────────────────────────────────────
    p_eval = sub.add_parser(
        "evaluate",
        help="Evaluate model on query/validation data.",
        description=(
            "Run the trained model over the Query_and_Validation_data.csv file and print "
            "overall accuracy, average confidence, low-confidence rate, and per-class accuracy. "
            "Predictions are written to the SQLite database so the monitoring dashboard can "
            "display them alongside live traffic."
        ),
        epilog=(
            "Examples:\n"
            "  pos-classifier evaluate\n"
            "  pos-classifier evaluate --data-dir path/to/data\n"
            "\n"
            "The command expects the following file to exist:\n"
            "  <data-dir>/Query_and_Validation_data.csv"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_eval.add_argument(
        "--data-dir", default="data", metavar="PATH",
        help="Directory containing Query_and_Validation_data.csv (default: data).",
    )

    # ── register ───────────────────────────────────────────────────────────────
    p_register = sub.add_parser(
        "register",
        help="Register a trained model to the MLflow Model Registry.",
        description=(
            "Link a completed MLflow training run to the Model Registry under the name "
            "'pos-classifier'. Once registered, the model version can be promoted through "
            "stages (Staging → Production) and loaded by the API server via --from-registry."
        ),
        epilog=(
            "Examples:\n"
            "  pos-classifier register --run-id abc123def456\n"
            "  pos-classifier register --run-id abc123def456 --description 'v2 with 5 epochs'\n"
            "\n"
            "Find the run ID in the MLflow UI (mlflow ui) or from the training log output.\n"
            "After registering, promote the version with:\n"
            "  pos-classifier transition --version <N> --stage Staging"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_register.add_argument(
        "--run-id", required=True, metavar="RUN_ID",
        help="MLflow run ID of the training run to register (shown in 'mlflow ui' or training logs).",
    )
    p_register.add_argument(
        "--description", metavar="TEXT",
        help="Free-text description attached to this model version (optional).",
    )

    # ── transition ─────────────────────────────────────────────────────────────
    p_transition = sub.add_parser(
        "transition",
        help="Promote or archive a registered model version.",
        description=(
            "Move a specific version of the 'pos-classifier' registered model to a new "
            "lifecycle stage. The API server's --from-registry flag loads whichever version "
            "is currently in the requested stage."
        ),
        epilog=(
            "Examples:\n"
            "  pos-classifier transition --version 3 --stage Staging\n"
            "  pos-classifier transition --version 3 --stage Production\n"
            "  pos-classifier transition --version 2 --stage Archived\n"
            "\n"
            "Stages:\n"
            "  Staging     — deployed for testing; served with --registry-stage Staging\n"
            "  Production  — live model; served by default with --from-registry\n"
            "  Archived    — retired; no longer served but kept for audit purposes"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_transition.add_argument(
        "--version", type=int, required=True, metavar="N",
        help="Integer version number to transition (shown by 'register' output or MLflow UI).",
    )
    p_transition.add_argument(
        "--stage", required=True, choices=["Staging", "Production", "Archived"],
        help="Target stage: Staging, Production, or Archived.",
    )

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
