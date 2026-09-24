"""
Road Damage Detector -- simple local demo.

Upload one road image, pick a model from the dropdown (Faster R-CNN,
YOLOv8n, YOLOv8n + Coordinate Attention, or YOLO26n) -> that checkpoint
runs inference -> the page shows boxes drawn on the image and a small
table of damage types and confidence scores. A separate "Compare all
models" mode runs all four on the same image side by side, adding an
EigenCAM heatmap + textual explanation for the three YOLO-family models
(model_registry.py explains why Faster R-CNN is excluded from that part).

Run locally:
    python demo/app.py
Then open http://127.0.0.1:7860
"""

import base64
import io

from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

import model_registry

app = Flask(__name__, static_folder="static", static_url_path="")


def _b64_image(pil_image: Image.Image) -> str:
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=90)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def _read_upload():
    if "image" not in request.files:
        return None, (jsonify({"error": "no image uploaded"}), 400)
    try:
        img = Image.open(request.files["image"].stream).convert("RGB")
    except Exception:
        return None, (jsonify({"error": "could not read image -- upload a JPG or PNG"}), 400)
    return img, None


def _read_conf():
    conf = float(request.form.get("confidence", 0.25))
    return min(max(conf, 0.01), 0.99)  # defensive clamp -- a bad client value must not reach the model call


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/models")
def models():
    return jsonify(model_registry.list_models())


@app.route("/api/predict", methods=["POST"])
def predict():
    img, err = _read_upload()
    if err:
        return err
    conf = _read_conf()
    model_id = request.form.get("model", "yolo26n")

    try:
        out = model_registry.predict_single(model_id, img, conf)
    except (ValueError, FileNotFoundError) as e:
        return jsonify({"error": str(e)}), 400

    return jsonify({
        "image": _b64_image(out["annotated"]),
        "detections": out["detections"],
        "counts": out["counts"],
        "latency_ms": out["latency_ms"],
    })


@app.route("/api/compare", methods=["POST"])
def compare():
    img, err = _read_upload()
    if err:
        return err
    conf = _read_conf()

    results = {}
    for spec in model_registry.list_models():
        model_id = spec["id"]
        try:
            if spec["supports_explain"]:
                out = model_registry.predict_with_explanation(model_id, img, conf)
                results[model_id] = {
                    "label": spec["label"],
                    "image": _b64_image(out["annotated"]),
                    "heatmap": _b64_image(out["heatmap"]),
                    "detections": out["detections"],
                    "counts": out["counts"],
                    "sentences": out["sentences"],
                    "latency_ms": out["latency_ms"],
                }
            else:
                out = model_registry.predict_single(model_id, img, conf)
                results[model_id] = {
                    "label": spec["label"],
                    "image": _b64_image(out["annotated"]),
                    "heatmap": None,
                    "detections": out["detections"],
                    "counts": out["counts"],
                    "sentences": [],
                    "latency_ms": out["latency_ms"],
                }
        except FileNotFoundError as e:
            results[model_id] = {"label": spec["label"], "error": str(e)}

    return jsonify(results)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7860, debug=False)
