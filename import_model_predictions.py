"""Import trained YOLO proposals into the annotation review queue."""

# pyright: reportMissingImports=false

import argparse
import json
import time
from pathlib import Path


ROOT = Path(__file__).parent
DEFAULT_MODEL = ROOT / "models" / "round1" / "weights" / "best.pt"
DEFAULT_IMAGES = ROOT / "training_images" / "collected"
DEFAULT_ANNOTATIONS = ROOT / "training_images" / "bounding_boxes.json"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
CLASSES = ("batman", "robin", "goon", "other_enemy")


def save_document(path: Path, document: dict[str, object], predicted: set[str]) -> None:
    document["version"] = 2
    document["classes"] = list(CLASSES)
    document["predicted_images"] = sorted(predicted)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--device", help='for example "0" for GPU or "cpu"')
    parser.add_argument(
        "--replace-predictions",
        action="store_true",
        help="rerun previous model predictions; manual annotations remain untouched",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if not arguments.model.is_file():
        raise SystemExit(f"Model not found: {arguments.model}")
    if not 0 < arguments.confidence < 1:
        raise SystemExit("--confidence must be between 0 and 1")
    print("Loading Ultralytics (the first import can take a little while)...", flush=True)
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("Install prediction support with: python -m pip install ultralytics") from error

    document = json.loads(arguments.annotations.read_text(encoding="utf-8"))
    annotations = document.setdefault("images", {})
    reviews = document.setdefault("reviews", {})
    predicted = set(map(str, document.get("predicted_images", [])))
    if not isinstance(annotations, dict) or not isinstance(reviews, dict):
        raise SystemExit("Invalid annotation document")

    candidates = sorted(
        path
        for path in arguments.images.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    selected = []
    for path in candidates:
        name = path.relative_to(arguments.images).as_posix()
        replaceable_prediction = (
            arguments.replace_predictions
            and name in predicted
            and reviews.get(name) not in {"correct", "fixed"}
        )
        if name not in annotations or replaceable_prediction:
            selected.append(path)
    if not selected:
        print("No unannotated images require predictions.")
        return

    print(f"Loading model: {arguments.model}", flush=True)
    model = YOLO(str(arguments.model))
    settings: dict[str, object] = {
        "conf": arguments.confidence,
        "imgsz": arguments.image_size,
        "verbose": False,
    }
    if arguments.device:
        settings["device"] = arguments.device

    completed = 0
    total_detections = 0
    started = time.monotonic()
    image_root = arguments.images.resolve()
    print(
        f"Predicting {len(selected)} images one at a time on "
        f"{arguments.device or 'the automatically selected device'}...",
        flush=True,
    )
    for path in selected:
        try:
            results = model.predict(source=str(path.resolve()), **settings)
        except RuntimeError as error:
            if "out of memory" in str(error).lower():
                raise SystemExit(
                    "CUDA ran out of memory on one image. Retry with "
                    "--image-size 512, or use --device cpu."
                ) from error
            raise
        result = results[0]
        name = Path(result.path).resolve().relative_to(image_root).as_posix()
        boxes = []
        if result.boxes is not None:
            normalized = result.boxes.xyxyn.cpu().tolist()
            class_ids = result.boxes.cls.cpu().tolist()
            confidences = result.boxes.conf.cpu().tolist()
            for coordinates, class_id, confidence in zip(normalized, class_ids, confidences):
                left, top, right, bottom = map(float, coordinates)
                class_name = str(result.names[int(class_id)])
                if class_name not in CLASSES:
                    continue
                boxes.append(
                    {
                        "class_name": class_name,
                        "x": round(left, 7),
                        "y": round(top, 7),
                        "width": round(right - left, 7),
                        "height": round(bottom - top, 7),
                        "confidence": round(float(confidence), 7),
                        "source": "model",
                    }
                )
        annotations[name] = boxes
        predicted.add(name)
        reviews.pop(name, None)
        completed += 1
        total_detections += len(boxes)
        if completed % 25 == 0:
            save_document(arguments.annotations, document, predicted)
        if completed == 1 or completed % 10 == 0 or completed == len(selected):
            elapsed = time.monotonic() - started
            rate = completed / max(elapsed, 0.001)
            remaining = (len(selected) - completed) / max(rate, 0.001)
            print(
                f"[{completed:>4}/{len(selected)}] "
                f"{completed / len(selected):>6.1%} | "
                f"{total_detections} boxes | "
                f"{rate:.1f} images/s | ETA {remaining / 60:.1f} min",
                flush=True,
            )

    save_document(arguments.annotations, document, predicted)
    print(f"Imported predictions for {completed} images into {arguments.annotations}")
    print("Start annotation_server.py and select 'Review model boxes'.")


if __name__ == "__main__":
    main()
