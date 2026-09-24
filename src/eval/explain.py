"""
Explainability -- EigenCAM visualizations for the proposed model (YOLO26n
only; CLAUDE.md scope decision, agreed explicitly rather than assumed).

WHY EIGENCAM AND NOT GRAD-CAM: classic Grad-CAM needs a single scalar class
score to backpropagate from, which is awkward for a detector's multi-box,
multi-scale output (which scalar? which of possibly several boxes?).
EigenCAM sidesteps this entirely -- it takes the principal component of the
target layer's raw activations via SVD, needs no backward pass and no
class/box selection, and is the standard choice used across existing
YOLO-CAM tooling for exactly this reason.

WHICH LAYER: layer 22 of the underlying nn.Sequential -- the last C3k2
block feeding the P5 (large-object) detection branch, immediately before
Detect. This is deep enough to carry semantic (not just edge/texture)
information but still spatial (H x W, not yet flattened), which is what
EigenCAM needs to place a heatmap back onto the image. Verified directly
against this project's own printed architecture (Chapter 5, §5.4): layer
22 is a 128-channel C3k2 output, one layer before the 3-scale Detect head.

Usage:
    python src/eval/explain.py --image data/India/train/images/India_001744.jpg
    python src/eval/explain.py --image path/to/img.jpg --out custom_name.png

Writes:
    experiments/results/figures/explainability/<image_stem>_eigencam.png
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = REPO_ROOT / "experiments" / "runs" / "yolo26n_source" / "weights" / "best.pt"
OUT_DIR = REPO_ROOT / "experiments" / "results" / "figures" / "explainability"

TARGET_LAYER_INDEX = 22  # last C3k2 before Detect -- see module docstring
IMGSZ = 640


def load_image(path: Path, imgsz: int = IMGSZ):
    """BGR->RGB, resize to a square imgsz, both as a display array (uint8,
    0-255) and a model-ready tensor (float32, 0-1, NCHW). No letterboxing --
    a plain resize is fine for a CAM overlay, which only needs to be
    visually aligned with what's on screen, not bit-exact with inference
    preprocessing."""
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise SystemExit(f"Could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (imgsz, imgsz))
    rgb_float = rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb_float).permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
    return rgb, rgb_float, tensor


class _SingleTensorOutput(nn.Module):
    """Wraps ultralytics' DetectionModel, whose eval-mode forward returns a
    (predictions_tensor, raw_dict) tuple, into something that returns a
    single plain tensor. pytorch-grad-cam's bookkeeping (irrelevant to
    EigenCAM's actual computation, which only reads the hooked target
    layer's activations captured during this same forward pass) calls
    `.cpu()` on whatever forward() returns and crashes on a bare tuple --
    this wrapper exists only to satisfy that, not to change what EigenCAM
    sees or computes."""

    def __init__(self, detection_model):
        super().__init__()
        self.detection_model = detection_model

    def forward(self, x):
        out = self.detection_model(x)
        return out[0] if isinstance(out, (tuple, list)) else out


CLASS_NAMES = {0: "D00 (longitudinal crack)", 1: "D10 (transverse crack)",
               2: "D20 (alligator crack)", 3: "D40 (pothole)"}


def _region_label(cx_frac: float, cy_frac: float) -> str:
    """Map a box center (as a fraction of image width/height) onto a plain
    3x3 grid description. Purely geometric -- no model output involved."""
    col = "left" if cx_frac < 1 / 3 else ("center" if cx_frac < 2 / 3 else "right")
    row = "upper" if cy_frac < 1 / 3 else ("middle" if cy_frac < 2 / 3 else "lower")
    if row == "middle" and col == "center":
        return "the center of the frame"
    return f"the {row}-{col} of the frame"


def describe_detection(box_xyxy, cls_id: int, conf: float, grayscale_cam: np.ndarray,
                        imgsz: int = IMGSZ) -> str:
    """Turn one detection into a sentence grounded ONLY in numbers actually
    computed here: the box's own EigenCAM statistics versus the image-wide
    average. No free-text generation, no guessing at causes the network
    never demonstrably used -- every clause below traces back to a printed
    number, which matters for a report/viva where fabricated reasoning is
    explicitly disallowed (CLAUDE.md rule 3)."""
    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, imgsz), min(y2, imgsz)

    box_region = grayscale_cam[y1:y2, x1:x2]
    if box_region.size == 0:
        box_region = np.array([0.0])

    box_mean = float(box_region.mean())
    image_mean = float(grayscale_cam.mean()) + 1e-8
    concentration = box_mean / image_mean  # >1 = box lit up more than average

    box_area_frac = ((x2 - x1) * (y2 - y1)) / (imgsz * imgsz)
    total_energy = float(grayscale_cam.sum()) + 1e-8
    box_energy_frac = float(box_region.sum()) / total_energy  # share of total "attention mass"

    cx_frac = (x1 + x2) / 2 / imgsz
    cy_frac = (y1 + y2) / 2 / imgsz
    region = _region_label(cx_frac, cy_frac)
    cls_name = CLASS_NAMES.get(cls_id, f"class {cls_id}")

    if concentration >= 1.5:
        grounding = (f"the model's activation inside this box is {concentration:.1f}x the image "
                     f"average, and the box alone ({box_area_frac * 100:.0f}% of the frame's area) "
                     f"accounts for {box_energy_frac * 100:.0f}% of its total attention -- "
                     f"the detection is well-localized to the damage itself")
    elif concentration >= 0.9:
        grounding = (f"activation inside this box is close to the image average "
                     f"({concentration:.1f}x), so the attention is not sharply concentrated on the "
                     f"box, though it is not being pulled away by another region either")
    else:
        grounding = (f"activation inside this box is only {concentration:.1f}x the image average -- "
                     f"the model's strongest attention in this frame lies mostly outside the box, "
                     f"so this detection is weakly grounded in what EigenCAM highlights")

    return (f"{cls_name}, confidence {conf:.2f}, in {region}: {grounding}.")


def run_eigencam(image_path: Path, out_path: Path, weights: Path = WEIGHTS,
                 target_layer_index: int = TARGET_LAYER_INDEX):
    from ultralytics import YOLO
    from pytorch_grad_cam import EigenCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image

    model = YOLO(str(weights))
    torch_model = model.model.eval()
    target_layer = torch_model.model[target_layer_index]
    wrapped_model = _SingleTensorOutput(torch_model).eval()

    rgb_uint8, rgb_float, tensor = load_image(image_path)

    cam = EigenCAM(model=wrapped_model, target_layers=[target_layer])
    # EigenCAM needs no class/box target (it never backpropagates -- see
    # module docstring), but this library version still requires the
    # `targets` argument to be passed explicitly as None.
    grayscale_cam = cam(tensor, targets=None)[0, :, :]  # (H, W) in [0, 1]
    overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

    # Draw the model's actual detection boxes on the same resized frame for
    # direct comparison -- "where it looked" next to "what it found".
    results = model.predict(rgb_uint8, conf=0.25, verbose=False)[0]
    annotated = results.plot()[:, :, ::-1]  # BGR -> RGB

    side_by_side = np.concatenate([annotated, overlay], axis=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR))

    sentences = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()
        sentences.append(describe_detection(xyxy, cls_id, conf, grayscale_cam))

    txt_path = out_path.with_suffix(".txt")
    txt_path.write_text(
        f"Textual explanation -- {image_path.name}\n"
        f"({len(sentences)} detection(s) at conf>=0.25)\n\n" +
        ("\n".join(f"- {s}" for s in sentences) if sentences else "No detections above threshold."),
        encoding="utf-8",
    )

    return out_path, txt_path, len(results.boxes), sentences


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="path to a road image")
    ap.add_argument("--out", default=None,
                    help="output filename (default: <image_stem>_eigencam.png "
                         "under experiments/results/figures/explainability/)")
    ap.add_argument("--weights", default=str(WEIGHTS))
    args = ap.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    out_name = args.out or f"{image_path.stem}_eigencam.png"
    out_path = OUT_DIR / out_name

    written, txt_written, n_dets, sentences = run_eigencam(
        image_path, out_path, weights=Path(args.weights))
    print(f"{image_path.name}: {n_dets} detection(s) at conf>=0.25")
    for s in sentences:
        print(f"  - {s}")
    print(f"Wrote {written} (left: detections, right: EigenCAM overlay)")
    print(f"Wrote {txt_written} (textual explanation)")


if __name__ == "__main__":
    main()
