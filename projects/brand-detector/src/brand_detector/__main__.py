"""CLI entry point: brand-detector train | serve | evaluate."""

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(prog="brand-detector")
    sub = parser.add_subparsers(dest="command")

    train_p = sub.add_parser("train", help="Fine-tune the model")
    train_p.add_argument("--epochs", type=int, default=5)

    serve_p = sub.add_parser("serve", help="Start the prediction API")
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8001)

    sub.add_parser("evaluate", help="Run evaluation on validation data")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "train":
        from brand_detector.training.trainer import train
        train()
    elif args.command == "serve":
        import uvicorn
        from brand_detector.api.app import create_app
        uvicorn.run(create_app(), host=args.host, port=args.port)
    elif args.command == "evaluate":
        from brand_detector.training.evaluate import evaluate
        evaluate()


if __name__ == "__main__":
    main()
