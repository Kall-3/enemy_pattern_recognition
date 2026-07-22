"""Import trained YOLO proposals into the annotation review queue."""

# pyright: reportMissingImports=false

import argparse
import json
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
        if name not in annotations or (arguments.replace_predictions and name in predicted):
            selected.append(path)
    if not selected:
        print("No unannotated images require predictions.")
        return

    model = YOLO(str(arguments.model))
    settings: dict[str, object] = {
        "source": [str(path.resolve()) for path in selected],
        "stream": True,
        "conf": arguments.confidence,
        "imgsz": arguments.image_size,
        "verbose": False,
    }
    if arguments.device:
        settings["device"] = arguments.device

    completed = 0
    image_root = arguments.images.resolve()
    for result in model.predict(**settings):
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
        if completed % 50 == 0:
            save_document(arguments.annotations, document, predicted)
            print(f"Predicted {completed}/{len(selected)} images")

    save_document(arguments.annotations, document, predicted)
    print(f"Imported predictions for {completed} images into {arguments.annotations}")
    print("Start annotation_server.py and select 'Review model boxes'.")


if __name__ == "__main__":
    main()
