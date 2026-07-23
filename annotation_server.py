"""Local web UI for drawing character bounding boxes on gameplay screenshots."""

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
DEFAULT_SCENE_LABELS = ROOT / "training_images" / "goon_labels.json"
DEFAULT_BOXES = ROOT / "training_images" / "bounding_boxes.json"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
CLASSES = ("batman", "robin", "goon", "other_enemy")

HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LEGO Batman box annotator</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, sans-serif; }
    * { box-sizing: border-box; }
    body { margin: 0; height: 100vh; overflow: hidden; background: #101114; color: #eee; }
    header { height: 54px; padding: 8px 14px; display: flex; gap: 14px;
             align-items: center; justify-content: space-between; background: #1a1c21; }
    #status { display: flex; gap: 14px; align-items: center; color: #bbb; }
    #scene-label { padding: 4px 9px; border-radius: 12px; background: #444; }
    #scene-label.yes { background: #176f38; } #scene-label.no { background: #7b2828; }
    #workspace { height: calc(100vh - 146px); display: grid; place-items: center;
                 padding: 8px; background: #090a0c; }
    #canvas-wrap { position: relative; max-width: 100%; max-height: 100%; }
    canvas { display: block; max-width: 100%; max-height: calc(100vh - 162px);
             width: auto; height: auto; cursor: crosshair; touch-action: none; }
    #empty { font-size: 22px; color: #888; }
    footer { height: 92px; display: flex; align-items: center; justify-content: center;
             gap: 8px; padding: 8px 12px; background: #1a1c21; }
    .classes { display: flex; gap: 6px; padding: 0 8px; border-left: 1px solid #444;
               border-right: 1px solid #444; }
    button { border: 2px solid transparent; border-radius: 7px; padding: 9px 13px;
             color: white; background: #454952; font-size: 14px; cursor: pointer; }
    button:hover { filter: brightness(1.15); }
    button.active { border-color: white; box-shadow: 0 0 0 2px #000; }
    button:disabled { opacity: .4; cursor: default; }
    select { padding: 10px; border-radius: 7px; background: #30333a; color: white; }
    .batman { background: #2574d8; } .robin { background: #28a850; }
    .goon { background: #c038d0; } .other_enemy { background: #e07828; }
    #save { background: #16723a; font-weight: 700; }
    #delete { background: #9c2b2b; }
    #dirty { width: 12px; height: 12px; border-radius: 50%; background: transparent; }
    #dirty.on { background: #ffd43b; }
    kbd { padding: 2px 5px; border: 1px solid #666; border-radius: 4px; background: #30333a; }
  </style>
</head>
<body>
  <header>
    <strong>Draw a tight box around each visible character</strong>
    <div id="status">
      <span id="filename"></span><span id="scene-label"></span>
      <span id="progress"></span><span id="dirty" title="Unsaved changes"></span>
    </div>
  </header>
  <main id="workspace">
    <div id="canvas-wrap"><canvas id="canvas"></canvas></div>
    <div id="empty" hidden>Every image has been reviewed.</div>
  </main>
  <footer>
    <button onclick="previousImage()">← Previous</button>
    <button onclick="skipImage()">Skip</button>
    <select id="queue-mode" onchange="changeQueue(this.value)">
      <option value="unboxed">Manual: unboxed</option>
      <option value="review">Review model boxes</option>
      <option value="needs_fix">Fix incorrect boxes</option>
      <option value="all">Browse all images</option>
    </select>
    <div class="classes" id="classes"></div>
    <button id="delete" onclick="deleteSelected()">Delete <kbd>Tab</kbd></button>
    <button id="correct" onclick="markReview('correct')" hidden>Correct <kbd>C</kbd></button>
    <button id="needs-fix" onclick="markReview('needs_fix')" hidden>Needs fixing <kbd>F</kbd></button>
    <button id="save" onclick="saveAndNext()">Save &amp; next <kbd>Space</kbd></button>
  </footer>
<script>
const CLASS_INFO = {
  batman: {label:'Batman', color:'#3287ff', key:'1'},
  robin: {label:'Robin', color:'#31d464', key:'2'},
  goon: {label:'Goon', color:'#ef4fff', key:'3'},
  other_enemy: {label:'Other enemy', color:'#ff902f', key:'4'}
};
const canvas = document.querySelector('#canvas');
const context = canvas.getContext('2d');
const background = new Image();
let current = null, boxes = [], selected = -1, activeClass = 'goon', queueMode = 'unboxed';
let gesture = null, dirty = false, history = [];

for (const [name, info] of Object.entries(CLASS_INFO)) {
  const button = document.createElement('button');
  button.className = name; button.dataset.className = name;
  button.textContent = `${info.key} · ${info.label}`;
  button.onclick = () => chooseClass(name);
  document.querySelector('#classes').appendChild(button);
}

function chooseClass(name) {
  activeClass = name;
  if (selected >= 0) {
    boxes[selected].class_name = name; boxes[selected].source = 'corrected'; setDirty(); draw();
  }
  document.querySelectorAll('[data-class-name]').forEach(button =>
    button.classList.toggle('active', button.dataset.className === name));
}

function setDirty(value=true) {
  dirty = value; document.querySelector('#dirty').classList.toggle('on', dirty);
}

function pointFromEvent(event) {
  const rect = canvas.getBoundingClientRect();
  return {x:(event.clientX-rect.left)*canvas.width/rect.width,
          y:(event.clientY-rect.top)*canvas.height/rect.height};
}

function pixelBox(box) {
  return {x:box.x*canvas.width, y:box.y*canvas.height,
          width:box.width*canvas.width, height:box.height*canvas.height};
}

function normalizedBox(box) {
  const result={class_name:box.class_name, x:box.x/canvas.width, y:box.y/canvas.height,
    width:box.width/canvas.width, height:box.height/canvas.height,
    source:box.source || 'manual'};
  if (box.confidence !== undefined) result.confidence=box.confidence;
  return result;
}

function handles(box) {
  return {nw:[box.x,box.y], ne:[box.x+box.width,box.y],
          sw:[box.x,box.y+box.height], se:[box.x+box.width,box.y+box.height]};
}

function handleAt(point, box) {
  const radius = 10 * canvas.width / Math.max(canvas.getBoundingClientRect().width, 1);
  for (const [name, position] of Object.entries(handles(box)))
    if (Math.hypot(point.x-position[0],point.y-position[1]) <= radius) return name;
  return null;
}

function boxAt(point) {
  for (let index=boxes.length-1; index>=0; index--) {
    const box=pixelBox(boxes[index]);
    if (point.x>=box.x && point.x<=box.x+box.width &&
        point.y>=box.y && point.y<=box.y+box.height) return index;
  }
  return -1;
}

function draw() {
  context.clearRect(0,0,canvas.width,canvas.height);
  if (background.complete && background.naturalWidth) context.drawImage(background,0,0);
  boxes.forEach((box,index) => {
    const pixel=pixelBox(box), info=CLASS_INFO[box.class_name] || CLASS_INFO.other_enemy;
    context.lineWidth = index===selected ? 4 : 2;
    context.setLineDash(box.source==='model' ? [9,6] : []);
    context.strokeStyle=info.color; context.strokeRect(pixel.x,pixel.y,pixel.width,pixel.height);
    context.setLineDash([]);
    context.font='bold 16px system-ui';
    const label=box.confidence === undefined ? info.label : `${info.label} ${Math.round(box.confidence*100)}%`;
    const labelWidth=context.measureText(label).width+10;
    context.fillStyle=info.color; context.fillRect(pixel.x,Math.max(0,pixel.y-23),labelWidth,23);
    context.fillStyle='#080808'; context.fillText(label,pixel.x+5,Math.max(17,pixel.y-6));
    if (index===selected) {
      context.fillStyle='#fff';
      for (const position of Object.values(handles(pixel)))
        context.fillRect(position[0]-5,position[1]-5,10,10);
    }
  });
}

canvas.addEventListener('pointerdown', event => {
  if (!current || queueMode==='review') return;
  const point=pointFromEvent(event); canvas.setPointerCapture(event.pointerId);
  if (selected>=0) {
    const resize=handleAt(point,pixelBox(boxes[selected]));
    if (resize) { gesture={type:'resize', handle:resize, start:point,
      original:{...pixelBox(boxes[selected])}}; return; }
  }
  const hit=boxAt(point);
  if (hit>=0) {
    selected=hit; activeClass=boxes[hit].class_name; chooseClass(activeClass);
    gesture={type:'move',start:point,original:{...pixelBox(boxes[hit])}}; draw(); return;
  }
  selected=-1;
  gesture={type:'create',start:point,current:point}; draw();
});

canvas.addEventListener('pointermove', event => {
  if (!gesture) return;
  const point=pointFromEvent(event), min=4;
  point.x=Math.max(0,Math.min(canvas.width,point.x));
  point.y=Math.max(0,Math.min(canvas.height,point.y));
  if (gesture.type==='create') {
    gesture.current=point; draw();
    const x=Math.min(gesture.start.x,point.x), y=Math.min(gesture.start.y,point.y);
    context.setLineDash([8,5]); context.strokeStyle=CLASS_INFO[activeClass].color;
    context.lineWidth=2; context.strokeRect(x,y,Math.abs(point.x-gesture.start.x),Math.abs(point.y-gesture.start.y));
    context.setLineDash([]); return;
  }
  const original=gesture.original;
  if (gesture.type==='move') {
    const x=Math.max(0,Math.min(canvas.width-original.width,original.x+point.x-gesture.start.x));
    const y=Math.max(0,Math.min(canvas.height-original.height,original.y+point.y-gesture.start.y));
    boxes[selected]=normalizedBox({...original,x,y,class_name:boxes[selected].class_name,
      source:'corrected'});
  } else {
    let left=original.x, top=original.y, right=original.x+original.width, bottom=original.y+original.height;
    if (gesture.handle.includes('w')) left=Math.min(point.x,right-min);
    if (gesture.handle.includes('e')) right=Math.max(point.x,left+min);
    if (gesture.handle.includes('n')) top=Math.min(point.y,bottom-min);
    if (gesture.handle.includes('s')) bottom=Math.max(point.y,top+min);
    boxes[selected]=normalizedBox({x:left,y:top,width:right-left,height:bottom-top,
      class_name:boxes[selected].class_name,source:'corrected'});
  }
  setDirty(); draw();
});

canvas.addEventListener('pointerup', event => {
  if (!gesture) return;
  if (gesture.type==='create') {
    const end=pointFromEvent(event), x=Math.max(0,Math.min(gesture.start.x,end.x));
    const y=Math.max(0,Math.min(gesture.start.y,end.y));
    const width=Math.min(canvas.width,Math.max(gesture.start.x,end.x))-x;
    const height=Math.min(canvas.height,Math.max(gesture.start.y,end.y))-y;
    if (width>=5 && height>=5) {
      boxes.push(normalizedBox({x,y,width,height,class_name:activeClass,source:'manual'}));
      selected=boxes.length-1; setDirty();
    }
  }
  gesture=null; draw();
});

function deleteSelected() {
  if (selected<0 || queueMode==='review') return;
  boxes.splice(selected,1); selected=-1; setDirty(); draw();
}

async function requestNext(after=null) {
  const parameters=new URLSearchParams({mode:queueMode});
  if (after) parameters.set('after',after);
  return await (await fetch('/api/next?'+parameters)).json();
}

async function showImage(name, pushHistory=true) {
  if (!name) {
    current=null; canvas.hidden=true; document.querySelector('#empty').hidden=false; return;
  }
  if (pushHistory && current) history.push(current);
  const parameters=new URLSearchParams({image:name,mode:queueMode});
  const data=await (await fetch('/api/item?'+parameters)).json();
  current=name; boxes=data.boxes || []; selected=-1; gesture=null; setDirty(false);
  document.querySelector('#filename').textContent=name;
  const badge=document.querySelector('#scene-label');
  badge.textContent=data.scene_label ? `Goon: ${data.scene_label}` : 'Unclassified';
  badge.className=data.scene_label || '';
  document.querySelector('#progress').textContent=
    `${data.queue_label} ${data.queue_position}/${data.queue_total}`;
  document.querySelector('#empty').hidden=true; canvas.hidden=false;
  background.onload=()=>{ canvas.width=background.naturalWidth; canvas.height=background.naturalHeight; draw(); };
  background.src='/image/'+encodeURIComponent(name)+'?v='+Date.now();
}

async function loadInitial() { const data=await requestNext(); await showImage(data.image,false); chooseClass(activeClass); }
async function skipImage() { if (dirty && !confirm('Discard unsaved box changes?')) return;
  const data=await requestNext(current); await showImage(data.image); }
async function previousImage() { if (!history.length) return;
  if (dirty && !confirm('Discard unsaved box changes?')) return;
  const name=history.pop(); await showImage(name,false); }
async function changeQueue(mode) {
  if (dirty && !confirm('Discard unsaved box changes?')) return;
  queueMode=mode; selected=-1;
  const reviewing=queueMode==='review';
  document.querySelector('#correct').hidden=!reviewing;
  document.querySelector('#needs-fix').hidden=!reviewing;
  document.querySelector('#delete').hidden=reviewing;
  document.querySelector('#save').hidden=reviewing;
  const data=await requestNext(); await showImage(data.image,false);
}
async function saveBoxes() {
  if (!current) return false;
  const response=await fetch('/api/boxes',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({image:current,boxes,
      review_status:queueMode==='needs_fix' ? 'fixed' : null})});
  if (!response.ok) { alert('Saving failed: '+await response.text()); return false; }
  setDirty(false); return true;
}
async function saveAndNext() { if (!await saveBoxes()) return;
  const data=await requestNext(current); await showImage(data.image); }
async function markReview(status) {
  if (!current || queueMode!=='review') return;
  const response=await fetch('/api/review',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({image:current,status})});
  if (!response.ok) { alert('Review save failed: '+await response.text()); return; }
  const data=await requestNext(current); await showImage(data.image);
}

document.addEventListener('keydown', event => {
  if (event.repeat) return;
  const byKey=Object.entries(CLASS_INFO).find(([,info])=>info.key===event.key);
  if (byKey) { chooseClass(byKey[0]); event.preventDefault(); }
  else if (event.key==='Tab') { deleteSelected(); event.preventDefault(); }
  else if (queueMode==='review' && event.key.toLowerCase()==='c') { markReview('correct'); event.preventDefault(); }
  else if (queueMode==='review' && event.key.toLowerCase()==='f') { markReview('needs_fix'); event.preventDefault(); }
  else if (event.code==='Space' && queueMode!=='review') { saveAndNext(); event.preventDefault(); }
  else if (event.key==='Escape') { selected=-1; gesture=null; draw(); }
});
window.addEventListener('beforeunload', event => { if (dirty) { event.preventDefault(); event.returnValue=''; } });
loadInitial();
</script></body></html>"""


class AnnotationStore:
    def __init__(self, image_directory: Path, scene_labels: Path, boxes_path: Path) -> None:
        self.image_directory = image_directory.resolve()
        self.boxes_path = boxes_path
        self.lock = threading.Lock()
        self.scene_labels: dict[str, str] = {}
        self.annotations: dict[str, list[dict[str, float | str]]] = {}
        self.reviews: dict[str, str] = {}
        self.predicted_images: set[str] = set()

        if scene_labels.exists():
            loaded = json.loads(scene_labels.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                self.scene_labels = {
                    str(name): str(label)
                    for name, label in loaded.items()
                    if label in {"yes", "no"}
                }

        if boxes_path.exists():
            loaded = json.loads(boxes_path.read_text(encoding="utf-8"))
            images = loaded.get("images", {}) if isinstance(loaded, dict) else {}
            if isinstance(images, dict):
                self.annotations = {
                    str(name): boxes
                    for name, boxes in images.items()
                    if isinstance(boxes, list)
                }
            reviews = loaded.get("reviews", {}) if isinstance(loaded, dict) else {}
            if isinstance(reviews, dict):
                self.reviews = {
                    str(name): str(status)
                    for name, status in reviews.items()
                    if status in {"correct", "needs_fix", "fixed"}
                }
            predicted = loaded.get("predicted_images", []) if isinstance(loaded, dict) else []
            if isinstance(predicted, list):
                self.predicted_images = {str(name) for name in predicted}

    def images(self) -> list[str]:
        images = [
            path.relative_to(self.image_directory).as_posix()
            for path in self.image_directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        return sorted(images, key=lambda name: (self.scene_labels.get(name) != "yes", name))

    def save(self) -> None:
        self.boxes_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.boxes_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "version": 2,
                    "classes": list(CLASSES),
                    "images": self.annotations,
                    "reviews": self.reviews,
                    "predicted_images": sorted(self.predicted_images),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.boxes_path)

    def validate_image(self, image: str) -> None:
        if image not in self.images():
            raise ValueError("Unknown image")

    def set_boxes(
        self,
        image: str,
        boxes: object,
        review_status: object = None,
    ) -> None:
        self.validate_image(image)
        if not isinstance(boxes, list):
            raise ValueError("Boxes must be a list")
        validated: list[dict[str, float | str]] = []
        for box in boxes:
            if not isinstance(box, dict) or box.get("class_name") not in CLASSES:
                raise ValueError("Invalid box class")
            try:
                x = float(box["x"])
                y = float(box["y"])
                width = float(box["width"])
                height = float(box["height"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("Invalid box coordinates") from error
            if not (
                0 <= x < 1
                and 0 <= y < 1
                and width > 0
                and height > 0
                and x + width <= 1.000001
                and y + height <= 1.000001
            ):
                raise ValueError("Box lies outside the image")
            validated.append(
                {
                    "class_name": str(box["class_name"]),
                    "x": round(x, 7),
                    "y": round(y, 7),
                    "width": round(width, 7),
                    "height": round(height, 7),
                    "source": (
                        str(box.get("source"))
                        if box.get("source") in {"model", "manual", "corrected"}
                        else "manual"
                    ),
                }
            )
            confidence = box.get("confidence")
            if confidence is not None:
                confidence_value = float(confidence)
                if not 0 <= confidence_value <= 1:
                    raise ValueError("Invalid confidence")
                validated[-1]["confidence"] = round(confidence_value, 7)
        with self.lock:
            self.annotations[image] = validated
            if any(box.get("source") == "model" for box in validated):
                self.predicted_images.add(image)
            if review_status == "fixed":
                self.reviews[image] = "fixed"
            self.save()

    def set_review(self, image: str, status: str) -> None:
        self.validate_image(image)
        if image not in self.annotations or status not in {"correct", "needs_fix"}:
            raise ValueError("Invalid review")
        with self.lock:
            self.reviews[image] = status
            self.save()

    def has_model_boxes(self, image: str) -> bool:
        return image in self.predicted_images


def make_handler(store: AnnotationStore) -> type[BaseHTTPRequestHandler]:
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
            query = parse_qs(parsed.query)
            if parsed.path == "/":
                self.send_bytes(HTML.encode(), "text/html; charset=utf-8")
                return
            if parsed.path == "/api/next":
                images = store.images()
                mode = query.get("mode", ["unboxed"])[0]
                if mode == "unboxed":
                    pending = [name for name in images if name not in store.annotations]
                elif mode == "review":
                    pending = [
                        name
                        for name in images
                        if store.has_model_boxes(name) and name not in store.reviews
                    ]
                elif mode == "needs_fix":
                    pending = [
                        name for name in images if store.reviews.get(name) == "needs_fix"
                    ]
                elif mode == "all":
                    pending = images
                else:
                    self.send_json({"error": "Unknown queue mode"}, 400)
                    return
                after = query.get("after", [None])[0]
                if after in pending and len(pending) > 1:
                    start = (pending.index(after) + 1) % len(pending)
                    pending = pending[start:] + pending[:start]
                self.send_json({"image": pending[0] if pending else None})
                return
            if parsed.path == "/api/item":
                image = query.get("image", [""])[0]
                mode = query.get("mode", ["unboxed"])[0]
                try:
                    store.validate_image(image)
                except ValueError:
                    self.send_json({"error": "Unknown image"}, 404)
                    return
                images = store.images()
                if mode == "review":
                    review_images = [name for name in images if store.has_model_boxes(name)]
                    reviewed = sum(name in store.reviews for name in review_images)
                    queue_total = len(review_images)
                    queue_position = reviewed + (0 if image in store.reviews else 1)
                    queue_label = "Review"
                elif mode == "needs_fix":
                    correction_images = [
                        name
                        for name in images
                        if store.reviews.get(name) in {"needs_fix", "fixed"}
                    ]
                    fixed = sum(store.reviews.get(name) == "fixed" for name in correction_images)
                    queue_total = len(correction_images)
                    queue_position = fixed + (0 if store.reviews.get(image) == "fixed" else 1)
                    queue_label = "Fix"
                elif mode == "all":
                    queue_total = len(images)
                    queue_position = images.index(image) + 1
                    queue_label = "Image"
                else:
                    queue_total = len(images)
                    annotated = len(store.annotations)
                    queue_position = annotated + (0 if image in store.annotations else 1)
                    queue_label = "Manual"
                self.send_json(
                    {
                        "image": image,
                        "boxes": store.annotations.get(image, []),
                        "scene_label": store.scene_labels.get(image),
                        "review_status": store.reviews.get(image),
                        "annotated": len(store.annotations),
                        "total": len(store.images()),
                        "queue_label": queue_label,
                        "queue_position": min(queue_position, max(queue_total, 1)),
                        "queue_total": queue_total,
                    }
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
            if self.path == "/api/boxes":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    payload = json.loads(self.rfile.read(length))
                    store.set_boxes(
                        str(payload["image"]),
                        payload["boxes"],
                        payload.get("review_status"),
                    )
                    self.send_json({"ok": True})
                except (KeyError, ValueError, json.JSONDecodeError) as error:
                    self.send_json({"error": str(error)}, 400)
                return
            if self.path == "/api/review":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    payload = json.loads(self.rfile.read(length))
                    store.set_review(str(payload["image"]), str(payload["status"]))
                    self.send_json({"ok": True})
                except (KeyError, ValueError, json.JSONDecodeError) as error:
                    self.send_json({"error": str(error)}, 400)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def log_message(self, format: str, *arguments: object) -> None:
            print(format % arguments)

    return Handler


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--scene-labels", type=Path, default=DEFAULT_SCENE_LABELS)
    parser.add_argument("--boxes", type=Path, default=DEFAULT_BOXES)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    arguments.images.mkdir(parents=True, exist_ok=True)
    store = AnnotationStore(arguments.images, arguments.scene_labels, arguments.boxes)
    server = ThreadingHTTPServer(("127.0.0.1", arguments.port), make_handler(store))
    address = f"http://127.0.0.1:{arguments.port}"
    print(f"Annotating images in {arguments.images.resolve()}")
    print(f"Bounding boxes will be saved to {arguments.boxes.resolve()}")
    print(f"Open {address} — press Ctrl+C here to stop")
    if not arguments.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(address,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped annotator.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
