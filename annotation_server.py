"""Local web UI for labeling screenshots that contain a Joker goon."""

import argparse
import json
import mimetypes
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


ROOT = Path(__file__).parent
DEFAULT_IMAGES = ROOT / "training_images" / "collected"
DEFAULT_LABELS = ROOT / "training_images" / "goon_labels.json"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Joker goon reviewer</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, sans-serif; }
    body { margin: 0; background: #111; color: #eee; text-align: center; }
    header { padding: 12px; display: flex; justify-content: space-between; }
    #frame { height: calc(100vh - 170px); display: grid; place-items: center; }
    img { max-width: 96vw; max-height: 100%; object-fit: contain; }
    button { border: 0; border-radius: 8px; padding: 14px 28px; margin: 8px;
             color: white; font-size: 18px; cursor: pointer; }
    #yes { background: #16833b; } #no { background: #b32d2d; }
    #skip, #undo { background: #555; }
    #empty { font-size: 24px; color: #aaa; }
    footer { color: #aaa; } kbd { background: #333; padding: 2px 6px; border-radius: 4px; }
  </style>
</head>
<body>
  <header><strong>Does this image contain a Joker goon?</strong><span id="progress"></span></header>
  <main id="frame"><img id="image" alt="Screenshot"><div id="empty" hidden>Everything is labeled.</div></main>
  <section>
    <button id="yes" onclick="label('yes')">Yes (Y)</button>
    <button id="no" onclick="label('no')">No (N)</button>
    <button id="skip" onclick="loadNext(true)">Skip (S)</button>
    <button id="undo" onclick="undo()">Undo (U)</button>
  </section>
  <footer id="name"></footer>
<script>
let current = null;
async function loadNext(skip=false) {
  const query = skip && current ? '?after=' + encodeURIComponent(current) : '';
  const data = await (await fetch('/api/next' + query)).json();
  current = data.image;
  document.querySelector('#progress').textContent = `${data.labeled}/${data.total} labeled`;
  document.querySelector('#image').hidden = !current;
  document.querySelector('#empty').hidden = !!current;
  document.querySelector('#name').textContent = current || '';
  if (current) document.querySelector('#image').src = '/image/' + encodeURIComponent(current) + '?v=' + Date.now();
}
async function label(value) {
  if (!current) return;
  await fetch('/api/label', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({image:current, label:value})});
  await loadNext();
}
async function undo() {
  const data = await (await fetch('/api/undo', {method:'POST'})).json();
  await loadNext();
  if (data.image) { current = data.image; document.querySelector('#image').hidden=false;
    document.querySelector('#empty').hidden=true; document.querySelector('#name').textContent=current;
    document.querySelector('#image').src='/image/'+encodeURIComponent(current)+'?v='+Date.now(); }
}
document.addEventListener('keydown', event => {
  if (event.repeat) return;
  if (event.key.toLowerCase()==='y') label('yes');
  else if (event.key.toLowerCase()==='n') label('no');
  else if (event.key.toLowerCase()==='s') loadNext(true);
  else if (event.key.toLowerCase()==='u') undo();
});
loadNext();
</script></body></html>"""


class LabelStore:
    def __init__(self, image_directory: Path, label_path: Path) -> None:
        self.image_directory = image_directory.resolve()
        self.label_path = label_path
        self.lock = threading.Lock()
        self.history: list[str] = []
        self.labels: dict[str, str] = {}
        if label_path.exists():
            loaded = json.loads(label_path.read_text(encoding="utf-8"))
            self.labels = {
                str(name): str(label)
                for name, label in loaded.items()
                if label in {"yes", "no"}
            }

    def images(self) -> list[str]:
        return sorted(
            path.relative_to(self.image_directory).as_posix()
            for path in self.image_directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    def save(self) -> None:
        self.label_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.label_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self.labels, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.label_path)

    def label(self, image: str, value: str) -> None:
        if image not in self.images() or value not in {"yes", "no"}:
            raise ValueError("Invalid image or label")
        with self.lock:
            self.labels[image] = value
            self.history.append(image)
            self.save()

    def undo(self) -> str | None:
        with self.lock:
            if not self.history:
                return None
            image = self.history.pop()
            self.labels.pop(image, None)
            self.save()
            return image


def make_handler(store: LabelStore) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, content: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def send_json(self, value: object, status: int = 200) -> None:
            self.send_bytes(json.dumps(value).encode(), "application/json", status)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self.send_bytes(HTML.encode(), "text/html; charset=utf-8")
                return
            if parsed.path == "/api/next":
                images = store.images()
                unlabeled = [name for name in images if name not in store.labels]
                after = parse_qs(parsed.query).get("after", [None])[0]
                if after in unlabeled and len(unlabeled) > 1:
                    start = (unlabeled.index(after) + 1) % len(unlabeled)
                    unlabeled = unlabeled[start:] + unlabeled[:start]
                image = unlabeled[0] if unlabeled else None
                self.send_json(
                    {"image": image, "labeled": len(store.labels), "total": len(images)}
                )
                return
            if parsed.path.startswith("/image/"):
                name = unquote(parsed.path.removeprefix("/image/"))
                path = (store.image_directory / name).resolve()
                if (
                    store.image_directory not in path.parents
                    or not path.is_file()
                    or path.suffix.lower() not in IMAGE_EXTENSIONS
                ):
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                self.send_bytes(path.read_bytes(), mime)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if self.path == "/api/label":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    payload = json.loads(self.rfile.read(length))
                    store.label(str(payload["image"]), str(payload["label"]))
                    self.send_json({"ok": True})
                except (KeyError, ValueError, json.JSONDecodeError):
                    self.send_json({"error": "Invalid label request"}, 400)
                return
            if self.path == "/api/undo":
                self.send_json({"image": store.undo()})
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def log_message(self, format: str, *arguments: object) -> None:
            print(format % arguments)

    return Handler


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    arguments.images.mkdir(parents=True, exist_ok=True)
    store = LabelStore(arguments.images, arguments.labels)
    server = ThreadingHTTPServer(("127.0.0.1", arguments.port), make_handler(store))
    address = f"http://127.0.0.1:{arguments.port}"
    print(f"Reviewing images in {arguments.images.resolve()}")
    print(f"Labels will be saved to {arguments.labels.resolve()}")
    print(f"Open {address} — press Ctrl+C here to stop")
    if not arguments.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(address,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped reviewer.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
