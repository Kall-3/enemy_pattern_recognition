"""Live template-matching preview for LEGO Batman enemies."""

# pyright: reportMissingImports=false

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mss
import numpy as np


TRAINING_DIRECTORY = Path(__file__).with_name("training_images")
TEMPLATE_DIRECTORY = TRAINING_DIRECTORY / "enemies"
SCALES = (0.75, 0.9, 1.0, 1.1, 1.25)


@dataclass(frozen=True)
class EnemyTemplate:
    name: str
    image: np.ndarray


@dataclass(frozen=True)
class Detection:
    x: int
    y: int
    width: int
    height: int
    confidence: float
    name: str


def load_templates(directory: Path) -> list[EnemyTemplate]:
    templates = []
    for path in sorted(directory.glob("**/*.png")):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Skipping unreadable image: {path}")
            continue
        if image.shape[0] < 8 or image.shape[1] < 8:
            print(f"Skipping template smaller than 8x8 pixels: {path}")
            continue
        # Use the containing directory as the class name, so pose images such as
        # goon_00.png and goon_01.png share one readable label.
        templates.append(EnemyTemplate(path.parent.name.replace("_", " "), image))
    return templates


def find_enemies(
    frame_gray: np.ndarray,
    templates: list[EnemyTemplate],
    threshold: float,
) -> list[Detection]:
    candidates = []
    frame_height, frame_width = frame_gray.shape

    for template in templates:
        for scale in SCALES:
            width = round(template.image.shape[1] * scale)
            height = round(template.image.shape[0] * scale)
            if width < 8 or height < 8 or width > frame_width or height > frame_height:
                continue

            resized = cv2.resize(
                template.image,
                (width, height),
                interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
            )
            scores = cv2.matchTemplate(
                frame_gray,
                resized,
                cv2.TM_CCOEFF_NORMED,
            )
            rows, columns = np.where(scores >= threshold)
            for y, x in zip(rows, columns):
                candidates.append(
                    Detection(
                        int(x),
                        int(y),
                        width,
                        height,
                        float(scores[y, x]),
                        template.name,
                    )
                )

    if not candidates:
        return []

    boxes = [[item.x, item.y, item.width, item.height] for item in candidates]
    confidences = [item.confidence for item in candidates]
    kept = cv2.dnn.NMSBoxes(boxes, confidences, threshold, 0.3)
    return [candidates[int(index)] for index in np.asarray(kept).reshape(-1)]


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> None:
    for detection in detections:
        top_left = (detection.x, detection.y)
        bottom_right = (
            detection.x + detection.width,
            detection.y + detection.height,
        )
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
        label = f"{detection.name} {detection.confidence:.0%}"
        label_y = max(20, detection.y - 8)
        cv2.putText(
            frame,
            label,
            (detection.x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.80,
        help="minimum match confidence from 0 to 1 (default: 0.80)",
    )
    parser.add_argument(
        "--monitor",
        type=int,
        default=1,
        help="monitor number to capture (default: 1)",
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="detect in one saved screenshot instead of capturing a monitor",
    )
    return parser.parse_args()


def detect_and_draw(
    frame: np.ndarray,
    templates: list[EnemyTemplate],
    threshold: float,
) -> list[Detection]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = find_enemies(gray, templates, threshold)
    draw_detections(frame, detections)
    return detections


def main() -> None:
    arguments = parse_arguments()
    if not 0.0 < arguments.threshold <= 1.0:
        raise SystemExit("--threshold must be greater than 0 and at most 1")

    templates = load_templates(TEMPLATE_DIRECTORY)
    if not templates:
        raise SystemExit(
            f"No enemy templates found. Add cropped PNG images to {TEMPLATE_DIRECTORY}"
        )

    print(f"Loaded {len(templates)} templates. Press Q in the preview to quit.")

    if arguments.image is not None:
        frame = cv2.imread(str(arguments.image), cv2.IMREAD_COLOR)
        if frame is None:
            raise SystemExit(f"Could not read screenshot: {arguments.image}")
        detections = detect_and_draw(frame, templates, arguments.threshold)
        print(f"Detected {len(detections)} enemies in {arguments.image}")
        cv2.imshow("LEGO Batman enemy detector", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    previous_time = time.perf_counter()

    with mss.mss() as capture:
        if arguments.monitor < 1 or arguments.monitor >= len(capture.monitors):
            raise SystemExit(
                f"Monitor {arguments.monitor} does not exist; "
                f"choose 1-{len(capture.monitors) - 1}"
            )
        monitor = capture.monitors[arguments.monitor]

        while True:
            frame = np.asarray(capture.grab(monitor))[:, :, :3].copy()
            detections = detect_and_draw(frame, templates, arguments.threshold)

            current_time = time.perf_counter()
            fps = 1.0 / max(current_time - previous_time, 0.0001)
            previous_time = current_time
            cv2.putText(
                frame,
                f"Enemies: {len(detections)}  FPS: {fps:.1f}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("LEGO Batman enemy detector", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
