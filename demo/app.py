"""
Road Damage Detector -- simple local demo.

Upload one road image -> the trained YOLO26n checkpoint runs inference ->
the page shows boxes drawn on the image and a small table of damage types
and confidence scores. No extra sections, no offline metrics -- just the
detector.

Run locally:
    python demo/app.py
Then open http://127.0.0.1:7860
"""

import base64
import io
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = REPO_ROOT / "experiments" / "runs" / "yolo26n_source" / "weights" / "best.pt"

CLASS_NAMES = {0: "D00 — Longitudinal crack", 1: "D10 — Transverse crack",
              2: "D20 — Alligator crack", 3: "D40 — Pothole"}

if not WEIGHTS.exists():
    raise SystemExit(
        f"Checkpoint not found at {WEIGHTS}.\n"
        f"Train Phase 3a first (src/models/train_yolo.py), or point WEIGHTS at "
        f"whatever the final chosen checkpoint is."
    )

from ultralytics import YOLO  # noqa: E402

MODEL = YOLO(str(WEIGHTS))

app = Flask(__name__, static_folder="static", static_url_path="")


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "no image uploaded"}), 400

    conf = float(request.form.get("confidence", 0.25))
    conf = min(max(conf, 0.01), 0.99)  # defensive clamp -- a bad client value must not reach the model call

    try:
        img = Image.open(request.files["image"].stream).convert("RGB")
    except Exception:
        return jsonify({"error": "could not read image -- upload a JPG or PNG"}), 400

    results = MODEL.predict(img, conf=conf, verbose=False)[0]

    # results.plot() draws via ultralytics' own annotator and returns BGR
    # (cv2 convention); flip to RGB before handing to PIL/the browser.
    annotated_rgb = results.plot()[:, :, ::-1]
    buf = io.BytesIO()
    Image.fromarray(annotated_rgb).save(buf, format="JPEG", quality=90)
    annotated_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    detections = []
    counts: dict[str, int] = {}
    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = CLASS_NAMES.get(cls_id, f"class {cls_id}")
        conf_score = float(box.conf[0])
        detections.append({"class": cls_name, "confidence": round(conf_score, 3)})
        counts[cls_name] = counts.get(cls_name, 0) + 1

    return jsonify({
        "image": f"data:image/jpeg;base64,{annotated_b64}",
        "detections": detections,
        "counts": counts,
        "latency_ms": round(results.speed.get("inference", 0.0), 1),
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7860, debug=False)
