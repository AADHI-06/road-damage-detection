# Road Damage Detector — Demo (Phase 7)

**Status: built and verified locally, browser end-to-end.** One card,
drag-and-drop, model dropdown, no marketing sections.

## What it does

Upload one road image, pick a model from the dropdown, and it runs
inference → boxes are drawn on the image, labeled with damage type
(D00/D10/D20/D40) and confidence, plus a small table listing every
detection.

**Four models available** (all trained checkpoints from this project —
see `experiment_log.csv`):

| Model | Checkpoint |
|---|---|
| Faster R-CNN (ResNet-50 FPN) | `experiments/runs/faster_rcnn_source/best.pt` |
| YOLOv8n | `experiments/runs/yolov8n_source/weights/best.pt` |
| YOLOv8n + Coordinate Attention (Phase 3b) | `experiments/runs/yolov8n_ca_source/weights/best.pt` |
| YOLO26n | `experiments/runs/yolo26n_source/weights/best.pt` |

A **"Compare all models"** button runs all four on the same upload and
shows them side by side. For the three YOLO-family models it also shows
an EigenCAM attention heatmap and a per-detection textual explanation
(same engine as `src/eval/explain.py`); Faster R-CNN is detections-only
in this view — its RPN/RoI-head architecture doesn't map onto the same
EigenCAM wiring the YOLO family shares (see `demo/model_registry.py` for
the reasoning). Detection counts in the compare view can differ slightly
from the single-model view for the YOLO-family models: the compare view
runs on the exact letterboxed frame the heatmap is drawn on, while the
single-model view uses ultralytics' own internal preprocessing — a
borderline, near-threshold detection can occasionally flip either way.
This is surfaced as a note in the UI, not hidden.

## Run it

```bash
pip install -r demo/requirements.txt
python demo/app.py
```

Open **http://127.0.0.1:7860**.

## Verified

Full upload → model dropdown → `/api/predict` → render flow tested
through the browser for all four models (India/Japan sample images), and
the `/api/compare` grid tested end-to-end (all four cards render, three
show heatmap + textual explanation, Faster R-CNN shows detections only).

## Files

```
demo/
├── app.py                Flask backend: exposes GET /api/models,
│                          POST /api/predict, POST /api/compare
├── model_registry.py      loads all four checkpoints, uniform predict/
│                          explain interface over them (RCNN vs. YOLO
│                          preprocessing differences live here)
├── static/
│   ├── index.html         the page (model dropdown, compare button)
│   ├── style.css
│   └── script.js           upload, drag-drop, model select, compare grid
└── requirements.txt        flask, ultralytics, torchvision, pillow
```

## Hosting

Not deployed — runs locally only (see project decision: GitHub + local
run is sufficient for a graded course project with a viva; no dependency
on a free-tier host staying up on demo day).
