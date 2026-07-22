"""Transparent Windows overlay that outlines detected LEGO Batman enemies."""

# pyright: reportMissingImports=false, reportAttributeAccessIssue=false

import argparse
import ctypes
import platform
import threading
import time
from ctypes import wintypes

import cv2
import mss
import numpy as np
from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget

from enemy_detector import Detection, TEMPLATE_DIRECTORY, find_enemies, load_templates


WINDOWS_DISPLAY_AFFINITY_EXCLUDE = 0x00000011
WINDOWS_EX_TRANSPARENT = 0x00000020
WINDOWS_EX_LAYERED = 0x00080000
WINDOWS_EX_TOOLWINDOW = 0x00000080
WINDOWS_EX_NOACTIVATE = 0x08000000
WINDOWS_GET_EXSTYLE = -20
WINDOWS_SET_EXSTYLE = -20


def find_game_client(title_fragment: str) -> tuple[int, dict[str, int]] | None:
    """Find a visible window and return its handle and client rectangle."""
    user32 = ctypes.windll.user32
    matches: list[int] = []
    callback_type = ctypes.WINFUNCTYPE(
        wintypes.BOOL,
        wintypes.HWND,
        wintypes.LPARAM,
    )

    @callback_type
    def visit_window(window: int, _parameter: int) -> bool:
        if not user32.IsWindowVisible(window):
            return True
        length = user32.GetWindowTextLengthW(window)
        if length <= 0:
            return True
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(window, buffer, length + 1)
        if title_fragment.casefold() in buffer.value.casefold():
            matches.append(window)
        return True

    user32.EnumWindows(visit_window, 0)
    if not matches:
        return None

    window = matches[0]
    client = wintypes.RECT()
    origin = wintypes.POINT(0, 0)
    if not user32.GetClientRect(window, ctypes.byref(client)):
        return None
    if not user32.ClientToScreen(window, ctypes.byref(origin)):
        return None
    width = client.right - client.left
    height = client.bottom - client.top
    if width <= 0 or height <= 0:
        return None
    return window, {
        "left": origin.x,
        "top": origin.y,
        "width": width,
        "height": height,
    }


class DetectionBridge(QObject):
    updated = Signal(object, object)
    game_missing = Signal()


class EnemyOverlay(QWidget):
    def __init__(self, bridge: DetectionBridge) -> None:
        super().__init__()
        self.detections: list[Detection] = []
        # Deliberately omit the game-title fragment so window discovery cannot
        # mistake the overlay itself for the game.
        self.setWindowTitle("Enemy detection overlay")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        bridge.updated.connect(self.show_detections)
        bridge.game_missing.connect(self.hide)

    def configure_windows_behavior(self) -> None:
        window = int(self.winId())
        user32 = ctypes.windll.user32
        style = user32.GetWindowLongW(window, WINDOWS_GET_EXSTYLE)
        style |= (
            WINDOWS_EX_TRANSPARENT
            | WINDOWS_EX_LAYERED
            | WINDOWS_EX_TOOLWINDOW
            | WINDOWS_EX_NOACTIVATE
        )
        user32.SetWindowLongW(window, WINDOWS_SET_EXSTYLE, style)
        if not user32.SetWindowDisplayAffinity(
            window,
            WINDOWS_DISPLAY_AFFINITY_EXCLUDE,
        ):
            print("Warning: Windows could not exclude the overlay from capture.")

    @Slot(object, object)
    def show_detections(
        self,
        rectangle: dict[str, int],
        detections: list[Detection],
    ) -> None:
        self.setGeometry(
            rectangle["left"],
            rectangle["top"],
            rectangle["width"],
            rectangle["height"],
        )
        self.detections = detections
        if not self.isVisible():
            self.show()
            self.configure_windows_behavior()
        self.update()

    def paintEvent(self, _event: object) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        for detection in self.detections:
            color = QColor(40, 255, 80)
            painter.setPen(QPen(color, 3))
            painter.drawRect(
                detection.x,
                detection.y,
                detection.width,
                detection.height,
            )
            label = f"{detection.name} {detection.confidence:.0%}"
            text_y = max(18, detection.y - 5)
            painter.setPen(QPen(QColor(0, 0, 0), 4))
            painter.drawText(detection.x, text_y, label)
            painter.setPen(QPen(color, 1))
            painter.drawText(detection.x, text_y, label)
        painter.end()


def detection_worker(
    bridge: DetectionBridge,
    stop_event: threading.Event,
    title: str,
    threshold: float,
    interval: float,
) -> None:
    templates = load_templates(TEMPLATE_DIRECTORY)
    if not templates:
        print(f"No enemy templates found in {TEMPLATE_DIRECTORY}")
        return

    missing_reported = False
    with mss.mss() as capture:
        while not stop_event.is_set():
            started = time.monotonic()
            game = find_game_client(title)
            if game is None:
                bridge.game_missing.emit()
                if not missing_reported:
                    print(f'Waiting for a visible window containing "{title}"...')
                    missing_reported = True
            else:
                missing_reported = False
                _window, rectangle = game
                frame = np.asarray(capture.grab(rectangle))[:, :, :3]
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detections = find_enemies(gray, templates, threshold)
                bridge.updated.emit(rectangle, detections)

            elapsed = time.monotonic() - started
            stop_event.wait(max(0.0, interval - elapsed))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--window-title",
        default="Batman",
        help='part of the game window title (default: "Batman")',
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.80,
        help="minimum match confidence from 0 to 1 (default: 0.80)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.20,
        help="minimum seconds between detection passes (default: 0.20)",
    )
    return parser.parse_args()


def main() -> None:
    if platform.system() != "Windows":
        raise SystemExit("The transparent overlay currently supports Windows only.")

    arguments = parse_arguments()
    if not 0.0 < arguments.threshold <= 1.0:
        raise SystemExit("--threshold must be greater than 0 and at most 1")
    if arguments.interval <= 0:
        raise SystemExit("--interval must be greater than zero")

    # Keep Qt overlay coordinates aligned with the physical pixels captured by MSS.
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except (AttributeError, OSError):
        pass

    application = QApplication([])
    bridge = DetectionBridge()
    overlay = EnemyOverlay(bridge)
    stop_event = threading.Event()
    worker = threading.Thread(
        target=detection_worker,
        args=(
            bridge,
            stop_event,
            arguments.window_title,
            arguments.threshold,
            arguments.interval,
        ),
        daemon=True,
    )
    application.aboutToQuit.connect(stop_event.set)
    worker.start()
    print("Overlay running. Close this terminal or press Ctrl+C to stop.")
    try:
        exit_code = application.exec()
    except KeyboardInterrupt:
        exit_code = 0
    finally:
        stop_event.set()
        worker.join(timeout=2)
        overlay.close()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
