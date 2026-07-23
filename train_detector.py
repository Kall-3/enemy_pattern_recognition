"""Train a small YOLO detector on the exported LEGO Batman dataset."""

# pyright: reportMissingImports=false

import argparse
from pathlib import Path


ROOT = Path(__file__).parent
DEFAULT_DATA = ROOT / "training_images" / "yolo_round1" / "dataset.yaml"
DEFAULT_PROJECT = ROOT / "models"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="epochs without validation improvement before stopping; 0 disables early stopping",
    )
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument(
        "--batch",
        type=int,
        default=-1,
        help="fixed integer batch size, or -1 for automatic selection",
    )
    parser.add_argument("--device", help='for example "0" for GPU or "cpu"')
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--name", default="round1")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if not arguments.data.is_file():
        raise SystemExit(f"Dataset config not found: {arguments.data}; run export_yolo_dataset.py")
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("Install training support with: python -m pip install ultralytics") from error

    model = YOLO(arguments.model)
    settings: dict[str, object] = {
        "data": str(arguments.data.resolve()),
        "epochs": arguments.epochs,
        "imgsz": arguments.image_size,
        "batch": arguments.batch,
        "project": str(arguments.project.resolve()),
        "name": arguments.name,
        "patience": arguments.patience,
        "workers": 0,
        "seed": 42,
        "deterministic": True,
        "plots": True,
        "exist_ok": True,
    }
    if arguments.device:
        settings["device"] = arguments.device
    results = model.train(**settings)
    print(f"Training output: {results.save_dir}")
    print(f"Best model: {Path(results.save_dir) / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
