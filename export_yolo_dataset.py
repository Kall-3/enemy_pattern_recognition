"""Export JSON bounding boxes as a grouped YOLO detection dataset."""

import argparse
import itertools
import json
import os
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).parent
DEFAULT_ANNOTATIONS = ROOT / "training_images" / "bounding_boxes.json"
DEFAULT_IMAGES = ROOT / "training_images" / "collected"
DEFAULT_OUTPUT = ROOT / "training_images" / "yolo_round1"
CLASSES = ("batman", "robin", "goon", "other_enemy")
TIMESTAMP = re.compile(r"gameplay_(\d{8}_\d{6})_\d+")


def capture_time(filename: str) -> datetime | None:
    match = TIMESTAMP.search(Path(filename).stem)
    return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S") if match else None


def make_capture_groups(names: list[str], maximum_gap: float) -> list[list[str]]:
    ordered = sorted(names, key=lambda name: (capture_time(name) or datetime.min, name))
    groups: list[list[str]] = []
    previous: datetime | None = None
    for name in ordered:
        current = capture_time(name)
        if not groups or current is None or previous is None:
            groups.append([name])
        elif (current - previous).total_seconds() > maximum_gap:
            groups.append([name])
        else:
            groups[-1].append(name)
        previous = current
    return groups


def choose_validation_groups(
    groups: list[list[str]],
    annotations: dict[str, list[dict[str, Any]]],
    fraction: float,
) -> set[int]:
    target = max(1, round(sum(map(len, groups)) * fraction))
    all_classes = {
        str(box["class_name"]) for boxes in annotations.values() for box in boxes
    }
    best_score: tuple[int, int, int] | None = None
    best_indices: set[int] = set()
    maximum_groups = min(len(groups) - 1, max(1, len(groups) // 2))
    for count in range(1, maximum_groups + 1):
        for combination in itertools.combinations(range(len(groups)), count):
            names = [name for index in combination for name in groups[index]]
            classes = {
                str(box["class_name"])
                for name in names
                for box in annotations[name]
            }
            score = (len(all_classes - classes), abs(len(names) - target), count)
            if best_score is None or score < best_score:
                best_score = score
                best_indices = set(combination)
    return best_indices


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def write_label(
    path: Path,
    boxes: list[dict[str, Any]],
    class_ids: dict[str, int],
) -> None:
    lines = []
    for box in boxes:
        class_name = str(box["class_name"])
        x, y = float(box["x"]), float(box["y"])
        width, height = float(box["width"]), float(box["height"])
        lines.append(
            f"{class_ids[class_name]} {x + width / 2:.7f} {y + height / 2:.7f} "
            f"{width:.7f} {height:.7f}"
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--group-gap", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if not 0.05 <= arguments.validation_fraction <= 0.5:
        raise SystemExit("--validation-fraction must be between 0.05 and 0.5")
    if arguments.output.exists():
        if not arguments.overwrite:
            raise SystemExit(f"Output already exists: {arguments.output} (use --overwrite)")
        shutil.rmtree(arguments.output)

    document = json.loads(arguments.annotations.read_text(encoding="utf-8"))
    raw = document.get("images", {})
    if not isinstance(raw, dict) or not raw:
        raise SystemExit("No bounding-box annotations found")
    annotations: dict[str, list[dict[str, Any]]] = {
        str(name): boxes for name, boxes in raw.items() if isinstance(boxes, list)
    }
    unknown = {
        str(box.get("class_name"))
        for boxes in annotations.values()
        for box in boxes
        if box.get("class_name") not in CLASSES
    }
    if unknown:
        raise SystemExit(f"Unknown classes: {', '.join(sorted(unknown))}")
    missing = [name for name in annotations if not (arguments.images / name).is_file()]
    if missing:
        raise SystemExit(f"Missing source image: {missing[0]}")

    groups = make_capture_groups(list(annotations), arguments.group_gap)
    validation_groups = choose_validation_groups(
        groups, annotations, arguments.validation_fraction
    )
    validation_names = {name for index in validation_groups for name in groups[index]}
    split_for = {
        name: "val" if name in validation_names else "train" for name in annotations
    }
    class_ids = {name: index for index, name in enumerate(CLASSES)}

    for split in ("train", "val"):
        (arguments.output / "images" / split).mkdir(parents=True, exist_ok=True)
        (arguments.output / "labels" / split).mkdir(parents=True, exist_ok=True)
    for name, boxes in annotations.items():
        split = split_for[name]
        source = arguments.images / name
        link_or_copy(source, arguments.output / "images" / split / source.name)
        write_label(
            arguments.output / "labels" / split / f"{source.stem}.txt",
            boxes,
            class_ids,
        )

    yaml_lines = [
        f"path: {json.dumps(arguments.output.resolve().as_posix())}",
        "train: images/train",
        "val: images/val",
        "names:",
        *(f"  {index}: {name}" for index, name in enumerate(CLASSES)),
    ]
    (arguments.output / "dataset.yaml").write_text(
        "\n".join(yaml_lines) + "\n", encoding="utf-8"
    )
    split_data = {
        "group_gap_seconds": arguments.group_gap,
        "train": sorted(name for name, split in split_for.items() if split == "train"),
        "val": sorted(validation_names),
    }
    (arguments.output / "split.json").write_text(
        json.dumps(split_data, indent=2) + "\n", encoding="utf-8"
    )

    for split in ("train", "val"):
        names = [name for name, assigned in split_for.items() if assigned == split]
        counts = Counter(
            str(box["class_name"]) for name in names for box in annotations[name]
        )
        print(f"{split}: {len(names)} images, {sum(counts.values())} boxes, {dict(counts)}")
    print(f"Capture groups kept intact: {len(groups)}")
    print(f"Dataset written to {arguments.output.resolve()}")


if __name__ == "__main__":
    main()
