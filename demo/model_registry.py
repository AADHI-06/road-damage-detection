"""
Loads all four trained detectors once at startup and exposes a uniform
predict/explain interface over them, so demo/app.py doesn't need to know
architecture-specific details (torchvision vs. ultralytics, label offsets,
EigenCAM target layers).

Model set (all four trained checkpoints from this project -- see
experiment_log.csv): Faster R-CNN (ResNet-50 FPN), YOLOv8n, YOLOv8n +
Coordinate Attention (Phase 3b), YOLO26n.

EigenCAM target layers were found by loading each checkpoint and printing
model.model.model (the underlying ultralytics nn.Sequential), not guessed:
YOLOv8n's last pre-Detect layer is index 21 (C2f); YOLO26n's is 22 (C3k2,
same as src/eval/explain.py's existing TARGET_LAYER_INDEX); YOLOv8n+CA's is
also 22, but for a different reason -- inserting the CoordinateAttention
module at the end of the backbone (CLAUDE.md Phase 3b) shifts every later
layer's index up by one relative to plain YOLOv8n.

Faster R-CNN has no EigenCAM support here (see the scope decision recorded
against this session's model-comparison feature): its RPN + RoI-head
architecture doesn't map onto the same "single spatial feature layer before
a detection head" structure the other three share, so extending EigenCAM to
it would need separate wiring, not just a different layer index.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS = REPO_ROOT / "experiments" / "runs"

sys.path.insert(0, str(REPO_ROOT / "src" / "eval"))
sys.path.insert(0, str(REPO_ROOT / "src" / "models"))

from attention import register_ca_module  # noqa: E402
from rcnn_model import build_faster_rcnn  # noqa: E402
from rcnn_dataset import LABEL_OFFSET  # noqa: E402
from explain import compute_eigencam, _letterbox, CLASS_NAMES as YOLO_CLASS_NAMES, IMGSZ  # noqa: E402

register_ca_module()  # must run before any YOLO(...) load, harmless for non-CA ones

RCNN_CLASS_NAMES = {0: "D00 (longitudinal crack)", 1: "D10 (transverse crack)",
                     2: "D20 (alligator crack)", 3: "D40 (pothole)"}

MODEL_SPECS = {
    "rcnn": {
        "label": "Faster R-CNN (ResNet-50 FPN)",
        "kind": "rcnn",
        "weights": RUNS / "faster_rcnn_source" / "best.pt",
        "supports_explain": False,
    },
    "yolov8n": {
        "label": "YOLOv8n",
        "kind": "yolo",
        "weights": RUNS / "yolov8n_source" / "weights" / "best.pt",
        "target_layer": 21,
        "supports_explain": True,
    },
    "yolov8n_ca": {
        "label": "YOLOv8n + Coordinate Attention",
        "kind": "yolo",
        "weights": RUNS / "yolov8n_ca_source" / "weights" / "best.pt",
        "target_layer": 22,
        "supports_explain": True,
    },
    "yolo26n": {
        "label": "YOLO26n",
        "kind": "yolo",
        "weights": RUNS / "yolo26n_source" / "weights" / "best.pt",
        "target_layer": 22,
        "supports_explain": True,
    },
}

_loaded = {}


def _load_yolo(path: Path):
    from ultralytics import YOLO
    return YOLO(str(path))


def _load_rcnn(path: Path):
    model = build_faster_rcnn(num_classes=len(RCNN_CLASS_NAMES) + 1, imgsz=IMGSZ,
                               pretrained=False, checkpoint=path)
    model.eval()
    return model


def get_model(model_id: str):
    """Lazy-load and cache. Lazy rather than eager at import time so a
    missing checkpoint only breaks the one model actually requested, not
    the whole app -- useful since not everyone training this project will
    necessarily have run every phase."""
    if model_id not in MODEL_SPECS:
        raise ValueError(f"Unknown model id: {model_id}")
    if model_id in _loaded:
        return _loaded[model_id]

    spec = MODEL_SPECS[model_id]
    if not spec["weights"].exists():
        raise FileNotFoundError(
            f"Checkpoint for '{model_id}' not found at {spec['weights']}. "
            f"Train it first (see CLAUDE.md phases) or remove it from MODEL_SPECS."
        )

    if spec["kind"] == "yolo":
        model = _load_yolo(spec["weights"])
    else:
        model = _load_rcnn(spec["weights"])

    _loaded[model_id] = model
    return model


def list_models():
    return [{"id": mid, "label": spec["label"], "supports_explain": spec["supports_explain"]}
            for mid, spec in MODEL_SPECS.items()]


def _draw_rcnn_boxes(image: Image.Image, boxes, labels, scores) -> Image.Image:
    """torchvision detection models have no built-in .plot() (unlike
    ultralytics), so boxes are drawn by hand here."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        name = RCNN_CLASS_NAMES.get(int(label), f"class {label}")
        draw.rectangle([x1, y1, x2, y2], outline="#e11d48", width=3)
        caption = f"{name.split(' ')[0]} {score:.2f}"
        text_bbox = draw.textbbox((x1, y1), caption)
        draw.rectangle([x1, y1 - (text_bbox[3] - text_bbox[1]) - 6, text_bbox[2] + 4, y1],
                        fill="#e11d48")
        draw.text((x1 + 2, y1 - (text_bbox[3] - text_bbox[1]) - 5), caption, fill="white")
    return annotated


def _predict_rcnn(model, image: Image.Image, conf: float):
    # Letterbox rather than squash-resize -- keeps non-square photos (e.g.
    # RDD2022's Norway subset) from being distorted, and matches the same
    # preprocessing used for the YOLO-family models in this demo.
    letterboxed = _letterbox(np.array(image.convert("RGB")), IMGSZ)
    resized = Image.fromarray(letterboxed)
    tensor = torch.from_numpy(letterboxed.astype(np.float32) / 255.0).permute(2, 0, 1)

    start = time.perf_counter()
    with torch.no_grad():
        output = model([tensor])[0]
    latency_ms = (time.perf_counter() - start) * 1000

    scores = output["scores"].numpy()
    keep = scores >= conf
    boxes = output["boxes"].numpy()[keep]
    labels = output["labels"].numpy()[keep] - LABEL_OFFSET
    scores = scores[keep]

    annotated = _draw_rcnn_boxes(resized, boxes, labels, scores)

    detections = []
    counts = {}
    for label, score in zip(labels, scores):
        name = RCNN_CLASS_NAMES.get(int(label), f"class {label}")
        detections.append({"class": name, "confidence": round(float(score), 3)})
        counts[name] = counts.get(name, 0) + 1

    return {"annotated": annotated, "detections": detections, "counts": counts,
            "latency_ms": round(latency_ms, 1)}


def _predict_yolo(model, image: Image.Image, conf: float):
    results = model.predict(image, conf=conf, verbose=False)[0]
    annotated_rgb = results.plot()[:, :, ::-1]
    annotated = Image.fromarray(annotated_rgb)

    detections = []
    counts = {}
    for box in results.boxes:
        cls_id = int(box.cls[0])
        name = YOLO_CLASS_NAMES.get(cls_id, f"class {cls_id}")
        score = float(box.conf[0])
        detections.append({"class": name, "confidence": round(score, 3)})
        counts[name] = counts.get(name, 0) + 1

    return {"annotated": annotated, "detections": detections, "counts": counts,
            "latency_ms": round(results.speed.get("inference", 0.0), 1)}


def predict_single(model_id: str, image: Image.Image, conf: float) -> dict:
    model = get_model(model_id)
    spec = MODEL_SPECS[model_id]
    if spec["kind"] == "rcnn":
        return _predict_rcnn(model, image, conf)
    return _predict_yolo(model, image, conf)


def predict_with_explanation(model_id: str, image: Image.Image, conf: float) -> dict:
    """Only valid for the three YOLO-family models (supports_explain=True).

    Runs prediction on the SAME letterboxed 640x640 array the EigenCAM
    heatmap is computed on (via compute_eigencam -> load_image), rather
    than ultralytics' own internal preprocessing (what predict_single uses)
    -- boxes and heatmap must share one coordinate frame for the overlay to
    align. This can occasionally flip a borderline, near-threshold
    detection in or out compared to predict_single on the identical image:
    expected, not a bug."""
    spec = MODEL_SPECS[model_id]
    if not spec["supports_explain"]:
        raise ValueError(f"'{model_id}' does not support EigenCAM explanation")
    model = get_model(model_id)
    out = compute_eigencam(image, model, spec["target_layer"], conf=conf)

    detections = []
    counts = {}
    for box in out["results"].boxes:
        cls_id = int(box.cls[0])
        name = YOLO_CLASS_NAMES.get(cls_id, f"class {cls_id}")
        score = float(box.conf[0])
        detections.append({"class": name, "confidence": round(score, 3)})
        counts[name] = counts.get(name, 0) + 1

    return {
        "annotated": Image.fromarray(out["annotated"]),
        "heatmap": Image.fromarray(out["overlay"]),
        "detections": detections,
        "counts": counts,
        "sentences": out["sentences"],
        "latency_ms": round(out["results"].speed.get("inference", 0.0), 1),
    }
