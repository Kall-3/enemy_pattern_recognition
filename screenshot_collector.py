"""Periodically capture gameplay screenshots for dataset collection."""

import argparse
import hashlib
import signal
import time
from datetime import datetime
from pathlib import Path

import mss
import mss.tools


DEFAULT_OUTPUT = Path(__file__).with_name("training_images") / "collected"
running = True


def stop_capture(_signal_number: int, _frame: object) -> None:
    global running
    running = False


def parse_region(value: str) -> dict[str, int]:
    try:
        left, top, width, height = (int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "region must be left,top,width,height"
        ) from error
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("region width and height must be positive")
    return {"left": left, "top": top, "width": width, "height": height}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="seconds between screenshots (default: 2)",
    )
    parser.add_argument(
        "--monitor",
        type=int,
        default=1,
        help="monitor number to capture (default: 1)",
    )
    parser.add_argument(
        "--region",
        type=parse_region,
        help="optional capture rectangle: left,top,width,height",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="stop after saving this many screenshots",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.interval <= 0:
        raise SystemExit("--interval must be greater than zero")
    if arguments.limit is not None and arguments.limit <= 0:
        raise SystemExit("--limit must be greater than zero")

    arguments.output.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGINT, stop_capture)
    signal.signal(signal.SIGTERM, stop_capture)
    saved = 0
    previous_hash = ""

    with mss.mss() as capture:
        if arguments.region is not None:
            area = arguments.region
        else:
            if arguments.monitor < 1 or arguments.monitor >= len(capture.monitors):
                raise SystemExit(
                    f"Monitor {arguments.monitor} does not exist; choose "
                    f"1-{len(capture.monitors) - 1}"
                )
            area = capture.monitors[arguments.monitor]

        print(f"Capturing {area} every {arguments.interval:g} seconds")
        print(f"Saving to {arguments.output.resolve()}")
        print("Press Ctrl+C to stop.")

        next_capture = time.monotonic()
        while running and (arguments.limit is None or saved < arguments.limit):
            screenshot = capture.grab(area)
            image_hash = hashlib.blake2b(screenshot.bgra, digest_size=16).hexdigest()
            if image_hash != previous_hash:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                path = arguments.output / f"gameplay_{timestamp}.png"
                mss.tools.to_png(screenshot.rgb, screenshot.size, output=str(path))
                saved += 1
                previous_hash = image_hash
                print(f"[{saved}] {path.name}")

            next_capture += arguments.interval
            time.sleep(max(0.0, next_capture - time.monotonic()))

    print(f"Stopped. Saved {saved} screenshots.")


if __name__ == "__main__":
    main()
