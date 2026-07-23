# LEGO Batman character detection

This project collects LEGO Batman gameplay screenshots, labels characters with
bounding boxes, trains a YOLO detector, and uses that model to propose boxes for
the remaining screenshots.

The detection classes are:

1. `batman`
2. `robin`
3. `goon`
4. `other_enemy`

## Setup on Windows

Open PowerShell in the project directory, create a virtual environment, and
install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For GPU training, install a CUDA-enabled PyTorch build separately. This
computer's GTX 1660 Super worked with the CUDA 12.8 package index. Check that
PyTorch can see it:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

Use `python` after activating `.venv`. Avoid `py script.py` if `py` selects a
different Python installation instead of the active environment.

Every script supports `--help`, for example:

```powershell
python .\train_detector.py --help
```

## The usual training cycle

Run these commands from the project directory.

### 1. Collect screenshots

```powershell
python .\screenshot_collector.py --interval 2
```

Play one or more maps while it runs. Press `Ctrl+C` to stop. Screenshots go to
`training_images\collected`.

### 2. Draw or review bounding boxes

```powershell
python .\annotation_server.py
```

The browser UI has four queues:

- **Manual: unboxed** — manually label images without boxes.
- **Review model boxes** — accept or reject model-generated boxes.
- **Fix incorrect boxes** — edit images previously marked as needing fixes.
- **Browse all images** — inspect any image.

Annotation controls:

| Action | Control |
|---|---|
| Select Batman | `1` |
| Select Robin | `2` |
| Select Goon | `3` |
| Select Other enemy | `4` |
| Draw a box | Drag over a character |
| Move a box | Drag inside it |
| Resize a box | Select it and drag a corner |
| Delete selected box | `Tab` |
| Save and move to next image | `Space` |
| Accept model boxes | `C` |
| Mark model boxes for correction | `F` |
| Cancel selection/current gesture | `Escape` |

Boxes are stored in `training_images\bounding_boxes.json`. Model predictions
are not trusted for training until they are marked **Correct** or manually fixed
and saved.

### 3. Export trusted annotations to YOLO format

```powershell
python .\export_yolo_dataset.py --overwrite
```

This exports:

- manually boxed images;
- model boxes marked **Correct**;
- model boxes that were corrected and saved.

It excludes unreviewed predictions and images still marked **Needs fixing**.
The generated dataset is `training_images\yolo_round1`.

Exporting can take time because it audits the annotations, groups nearby
screenshots to reduce train/validation leakage, selects validation groups, and
hard-links or copies every selected image while creating a YOLO label file.

### 4. Train a model

Recommended YOLO11 Small command for the GTX 1660 Super:

```powershell
python .\train_detector.py --model yolo11s.pt --device 0 --batch 4 --epochs 100 --name small_round1
```

If CUDA runs out of memory, use `--batch 2`. The useful outputs are:

```text
models\small_round1\weights\best.pt
models\small_round1\weights\last.pt
models\small_round1\results.csv
models\small_round1\results.png
```

- `best.pt` is the checkpoint with the best validation fitness and should
  normally be used for predictions.
- `last.pt` is the latest training state and is used to resume an interrupted
  run.
- `results.csv` contains the training and validation metrics for each epoch.

Do not leave `results.csv` open in Excel while training. Excel can lock the file
and stop the trainer when it tries to append the next epoch.

Resume an interrupted run:

```powershell
python .\train_detector.py --resume .\models\small_round1\weights\last.pt --device 0
```

Resume continues toward the run's original epoch target; it does not add a new
set of epochs. Use `last.pt` for an exact continuation.

### 5. Predict boxes for unannotated images

```powershell
python .\import_model_predictions.py --model .\models\small_round1\weights\best.pt --device 0
```

The script processes one image at a time, prints progress and an ETA, and saves
a checkpoint every 25 images.

After training a newer model, replace predictions that have not been accepted
or fixed:

```powershell
python .\import_model_predictions.py --model .\models\small_round2\weights\best.pt --device 0 --replace-predictions
```

`--replace-predictions` preserves manual annotations and reviewed/fixed work.

### 6. Review, fix, and repeat

Start `annotation_server.py` again:

1. Use **Review model boxes** and press `C` or `F`.
2. Use **Fix incorrect boxes**, correct each image, and press `Space`.
3. Export again with `--overwrite`.
4. Train the next round under a new name such as `small_round2`.
5. Import the new predictions with `--replace-predictions`.

## Script reference

### `screenshot_collector.py`

Periodically captures a monitor or rectangular region. Identical consecutive
frames are skipped.

Useful options:

```text
--interval SECONDS
--monitor NUMBER
--region LEFT,TOP,WIDTH,HEIGHT
--output DIRECTORY
--limit COUNT
```

Example capturing 500 images from part of the screen:

```powershell
python .\screenshot_collector.py --interval 1 --region 100,100,1280,720 --limit 500
```

### `annotation_server.py`

Runs the local bounding-box browser interface. It only listens on
`127.0.0.1`, so it is not exposed to other computers.

Useful options:

```text
--images DIRECTORY
--scene-labels FILE
--boxes FILE
--port NUMBER
--no-browser
```

Example using port 8766:

```powershell
python .\annotation_server.py --port 8766
```

### `export_yolo_dataset.py`

Converts trusted JSON annotations into YOLO image/label folders and writes
`dataset.yaml`.

Useful options:

```text
--annotations FILE
--images DIRECTORY
--output DIRECTORY
--validation-fraction FRACTION
--group-gap SECONDS
--overwrite
```

`--group-gap` keeps screenshots captured close together in the same dataset
split. This reduces the chance of almost identical frames appearing in both
training and validation data.

### `train_detector.py`

Trains an Ultralytics YOLO object detector.

Useful options:

```text
--data DATASET_YAML
--model MODEL_OR_CHECKPOINT
--epochs COUNT
--patience COUNT
--image-size PIXELS
--batch COUNT
--device 0|cpu
--project DIRECTORY
--name RUN_NAME
--resume PATH_TO_LAST.PT
```

`--patience 0` disables early stopping and is the default. During a normal new
training run, `--model` may be a base model such as `yolo11s.pt` or an existing
`best.pt` checkpoint for fine-tuning.

### `import_model_predictions.py`

Runs a trained YOLO model over images that do not yet have annotations and adds
its boxes to the review queue.

Useful options:

```text
--model PATH_TO_BEST.PT
--images DIRECTORY
--annotations FILE
--confidence 0.25
--image-size 640
--device 0|cpu
--replace-predictions
```

Lower confidence includes more uncertain boxes; higher confidence includes
fewer, more certain boxes. If a 6 GB GPU runs out of memory, try
`--image-size 512`.

### `virtual_joystick.py`

Creates a virtual controller and maps global keyboard shortcuts to its left
stick. It uses vJoy through `pyvjoy` on Windows and `evdev` on Linux.

Global controls:

| Action | Shortcut |
|---|---|
| Walk | `Ctrl+Alt+Arrow` |
| Walk diagonally | Hold two arrow keys |
| Sprint | Add `Shift` while moving |
| Controller button 1 | `Ctrl+Alt+J` |
| Stop the script | `Ctrl+Alt+Q` |

Walk strength is `0.55`; sprint strength is `1.0`. The shortcuts work globally,
including while the game is focused, as long as the operating system and game
allow the keyboard listener to receive them.

The reusable movement functions in the file are:

```python
move_left(strength=0.55)
move_right(strength=0.55)
move_up(strength=0.55)
move_down(strength=0.55)
stop_moving()
set_stick(x, y)
set_button(button, pressed)
```

### `enemy_detector.py` (older experiment)

Uses OpenCV template matching, not the trained YOLO model. It loads cropped PNG
templates from `training_images\enemies` and displays detections in a separate
preview window.

```powershell
python .\enemy_detector.py --monitor 1 --threshold 0.8
```

Press `Q` in the preview to quit. It can also inspect one saved image:

```powershell
python .\enemy_detector.py --image .\example.png
```

### `enemy_overlay.py` (older experiment)

Displays the template matcher's boxes as a transparent, click-through Windows
overlay positioned over a game window. It currently uses `enemy_detector.py`;
it does **not** use `best.pt`.

```powershell
python .\enemy_overlay.py --window-title Batman --threshold 0.8
```

The title only needs to be a substring of the actual window title, so `Batman`
also matches a title containing LEGO® Batman™.

## Project data

```text
training_images/
├── collected/             captured screenshots
├── bounding_boxes.json    annotations, predictions, and review states
├── goon_labels.json       older scene-level yes/no labels
└── yolo_round1/           generated YOLO dataset

models/
└── RUN_NAME/
    ├── weights/best.pt
    ├── weights/last.pt
    ├── results.csv
    └── training plots
```

The large captured images, generated YOLO datasets, virtual environment, caches,
and model outputs are ignored by Git. Store/share the large training dataset
through the Hugging Face dataset repository instead of committing it to Git.
