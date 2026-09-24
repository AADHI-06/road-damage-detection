"""
Single-model, single-dataset evaluation.

Runs a trained YOLOv8 or Faster R-CNN over a directory of images, converts both
models' outputs into one common format, and scores them with src/eval/metrics.py.
Because both paths end in the same metric code, YOLOv8 and Faster R-CNN numbers
are directly comparable (see the header of metrics.py for why that matters).

Also measures inference latency and FPS at batch size 1 (the project spec section 5.3),
which is the setting that reflects per-image deployment cost.

Usage:
    python src/eval/evaluate.py --model yolo \
        --weights experiments/runs/yolov8n_source/weights/best.pt \
        --images data/processed/source/images/test \
        --labels data/processed/source/labels/test \
        --split test --dataset source

    python src/eval/evaluate.py --model faster_rcnn \
        --weights experiments/runs/faster_rcnn_source/best.pt ...
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "eval"))
sys.path.insert(0, str(REPO_ROOT / "src" / "models"))
sys.path.insert(0, str(REPO_ROOT / "src" / "utils"))

from metrics import evaluate_detections, CLASS_NAMES  # noqa: E402
from attention import register_ca_module  # noqa: E402

# Must run before any YOLO(weights) load below (predict_yolo, cross_country_eval.py's
# per-model parameter count) -- harmless no-op for a checkpoint that isn't the
# Coordinate Attention variant. Idempotent (see attention.py).
register_ca_module()

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def load_hyperparams() -> dict:
    with open(REPO_ROOT / "config" / "hyperparams.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_images(images_dir: Path) -> list[Path]:
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def labels_dir_for(images_dir: Path) -> Path:
    """Map .../images/<split> to .../labels/<split>.

    Done on path COMPONENTS, not by string replacement. str(Path) renders
    separators as backslashes on Windows, so a naive
    str(p).replace("/images/", "/labels/") silently does nothing there and
    evaluation ends up looking for labels inside the image directory --
    scoring every model against zero ground truth and reporting mAP 0.0000.
    That bug is invisible unless ground-truth counts are printed, so the
    component-wise form below is used instead.
    """
    parts = list(Path(images_dir).parts)
    if "images" not in parts:
        raise ValueError(f"Expected an 'images' component in {images_dir}")
    # Replace the LAST 'images' component, in case a parent dir is also named that.
    idx = len(parts) - 1 - parts[::-1].index("images")
    parts[idx] = "labels"
    return Path(*parts)


def load_ground_truth(image_paths: list[Path], labels_dir: Path) -> list[dict]:
    """Read YOLO labels and convert to absolute xyxy, matching prediction space.

    Uses PIL to read each image's real dimensions rather than trusting the
    original XML <size> tag, so de-normalization can never disagree with the
    pixels the model actually saw.
    """
    from PIL import Image

    gts = []
    for image_path in image_paths:
        with Image.open(image_path) as img:
            width, height = img.size

        label_path = labels_dir / f"{image_path.stem}.txt"
        boxes, labels = [], []
        if label_path.exists():
            text = label_path.read_text(encoding="utf-8").strip()
            for line in text.splitlines() if text else []:
                class_id, xc, yc, bw, bh = line.split()
                xc, yc, bw, bh = float(xc), float(yc), float(bw), float(bh)
                boxes.append([
                    (xc - bw / 2) * width,
                    (yc - bh / 2) * height,
                    (xc + bw / 2) * width,
                    (yc + bh / 2) * height,
                ])
                labels.append(int(class_id))

        gts.append({
            "boxes": np.array(boxes, dtype=np.float64) if boxes else np.zeros((0, 4)),
            "labels": np.array(labels, dtype=int) if labels else np.zeros(0, dtype=int),
        })
    return gts


def predict_yolo(weights: Path, image_paths: list[Path], imgsz: int,
                 min_conf: float, max_det: int, device: str = "cpu") -> list[dict]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    preds = []
    # Predict in chunks to bound memory; ultralytics returns boxes already mapped
    # back to each image's original pixel coordinates.
    chunk = 32
    for start in range(0, len(image_paths), chunk):
        batch = [str(p) for p in image_paths[start:start + chunk]]
        results = model.predict(batch, imgsz=imgsz, conf=min_conf, max_det=max_det,
                                verbose=False, device=device)
        for r in results:
            b = r.boxes
            if b is None or len(b) == 0:
                preds.append({"boxes": np.zeros((0, 4)), "scores": np.zeros(0),
                              "labels": np.zeros(0, dtype=int)})
            else:
                preds.append({
                    "boxes": b.xyxy.cpu().numpy().astype(np.float64),
                    "scores": b.conf.cpu().numpy().astype(np.float64),
                    "labels": b.cls.cpu().numpy().astype(int),
                })
    return preds


def predict_faster_rcnn(weights: Path, image_paths: list[Path], imgsz: int,
                        min_conf: float, max_det: int, device: str = "cpu") -> list[dict]:
    import torch
    from rcnn_dataset import LABEL_OFFSET
    from rcnn_model import build_faster_rcnn
    from PIL import Image

    torch_device = torch.device(device)

    # Same builder the training script uses, so the evaluated architecture
    # cannot drift from the trained one.
    model = build_faster_rcnn(len(CLASS_NAMES) + 1, imgsz, pretrained=False,
                              checkpoint=weights)
    model.eval()
    model.to(torch_device)

    preds = []
    with torch.no_grad():
        for image_path in image_paths:
            img = Image.open(image_path).convert("RGB")
            tensor = torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1).float() / 255.0
            tensor = tensor.to(torch_device)
            out = model([tensor])[0]

            scores = out["scores"].cpu().numpy().astype(np.float64)
            keep = scores >= min_conf
            boxes = out["boxes"].cpu().numpy().astype(np.float64)[keep]
            # Shift torchvision's 1..4 labels back to the project's canonical 0..3.
            labels = out["labels"].cpu().numpy().astype(int)[keep] - LABEL_OFFSET
            scores = scores[keep]

            # Cap detections per image to match the YOLO path's max_det, so neither
            # model gets an unfair recall advantage from emitting more boxes.
            if len(scores) > max_det:
                order = np.argsort(-scores)[:max_det]
                boxes, scores, labels = boxes[order], scores[order], labels[order]

            preds.append({"boxes": boxes, "scores": scores, "labels": labels})
    return preds


def get_predictions_and_gt(model_kind: str, weights: Path, images_dir: Path,
                           labels_dir: Path, hp: dict,
                           limit: int | None = None,
                           sample_size: int | None = None, sample_seed: int = 42,
                           device: str = "cpu") -> tuple[list[Path], list[dict], list[dict]]:
    """Run inference and load matching ground truth. Pure data, no scoring.

    Factored out of run_evaluation so a caller that also wants per-image detail
    (cross_country_eval.py's qualitative failure examples) can get it from the
    SAME inference pass instead of re-running the model a second time -- Faster
    R-CNN inference is expensive enough that this matters (it's ~0.9s/image on
    CPU, so evaluating twice would double a multi-hour run for no reason).

    `limit` slices image_paths BEFORE inference runs, not after -- slicing
    after would still pay the full-dataset inference cost and only trim the
    result, defeating the entire point of a fast pipeline check. Used by
    --smoke; deterministic (first N) is fine there since it's a pipeline
    check, not a result.

    `sample_size` is different: a SEEDED RANDOM subsample, for a dataset
    that's genuinely too large to evaluate in full within the available
    compute (e.g. Norway's 8,161 images). A plain slice would be biased --
    RDD2022 filenames are capture-sequence ordered, so "first N" would
    cluster toward one drive route/session instead of being representative.
    random.Random(sample_seed).sample() avoids that, and the seed makes the
    subsample reproducible across reruns.

    `device`: forwarded to the actual inference calls. Previously this
    function silently ignored whatever device the caller asked for --
    predict_yolo hardcoded "cpu" and predict_faster_rcnn never moved the
    model off it, so passing --device cuda anywhere upstream had no effect
    on the (expensive) bulk inference here, only on the separate latency
    micro-benchmark in benchmark_speed.py. Fixed; do not reintroduce a path
    that drops `device` on the way to predict_yolo/predict_faster_rcnn.
    """
    ev = hp["eval"]
    imgsz = hp["yolo"]["imgsz"] if model_kind == "yolo" else hp["faster_rcnn"]["imgsz"]

    image_paths = list_images(images_dir)
    if sample_size is not None and sample_size < len(image_paths):
        import random

        image_paths = sorted(random.Random(sample_seed).sample(image_paths, sample_size))
    elif limit is not None:
        image_paths = image_paths[:limit]
    gts = load_ground_truth(image_paths, labels_dir)

    if model_kind == "yolo":
        preds = predict_yolo(weights, image_paths, imgsz,
                             ev["min_confidence"], ev["max_detections_per_image"], device=device)
    else:
        preds = predict_faster_rcnn(weights, image_paths, imgsz,
                                    ev["min_confidence"], ev["max_detections_per_image"], device=device)
    return image_paths, preds, gts


def assert_has_ground_truth(gts: list[dict], labels_dir: Path, num_images: int) -> int:
    """Fail loudly on empty ground truth. Every metric is mathematically 0.0 in
    that case, which looks like a catastrophically bad model rather than the
    data-loading fault it almost always is (wrong labels dir, bad path
    derivation, unconverted split). Never let that reach a results table.
    Returns the total box count so callers can log/print it.
    """
    total_gt_boxes = sum(len(g["labels"]) for g in gts)
    if total_gt_boxes == 0:
        raise SystemExit(
            f"No ground-truth boxes loaded from {labels_dir}\n"
            f"Checked {num_images} images. Metrics would all be 0.0 and meaningless.\n"
            f"Verify the labels directory exists and contains .txt files matching the image stems."
        )
    return total_gt_boxes


def run_evaluation(model_kind: str, weights: Path, images_dir: Path, labels_dir: Path,
                   hp: dict, measure_speed: bool = True, device: str = "cpu") -> dict:
    ev = hp["eval"]
    imgsz = hp["yolo"]["imgsz"] if model_kind == "yolo" else hp["faster_rcnn"]["imgsz"]

    print(f"Evaluating {model_kind} on images from {images_dir}")

    t0 = time.perf_counter()
    image_paths, preds, gts = get_predictions_and_gt(
        model_kind, weights, images_dir, labels_dir, hp, device=device)
    eval_runtime = time.perf_counter() - t0

    total_gt_boxes = assert_has_ground_truth(gts, labels_dir, len(image_paths))
    print(f"  {len(image_paths)} images, loaded {total_gt_boxes} ground-truth boxes from {labels_dir}")

    results = evaluate_detections(
        preds, gts,
        num_classes=len(CLASS_NAMES),
        iou_start=ev["iou_thresholds_start"],
        iou_end=ev["iou_thresholds_end"],
        iou_step=ev["iou_thresholds_step"],
        pr_iou_threshold=ev["pr_iou_threshold"],
        pr_confidence_threshold=ev["pr_confidence_threshold"],
    )

    # Speed and parameter count come from src/eval/benchmark_speed.py so there is
    # exactly one definition of how they are measured (the project spec section 5.3).
    from benchmark_speed import measure_latency

    if measure_speed:
        speed = measure_latency(model_kind, weights, image_paths, imgsz, device)
    else:
        # Parameter count is architecture-only and cheap, so it is still recorded
        # even when timing is skipped.
        from benchmark_speed import load_model, count_parameters

        _, module = load_model(model_kind, weights, imgsz, device)
        speed = {**count_parameters(module),
                 "latency_ms": float("nan"), "fps": float("nan")}

    latency_ms, fps = speed["latency_ms"], speed["fps"]

    results.update({
        "num_images": len(image_paths),
        "eval_runtime_sec": round(eval_runtime, 2),
        "latency_ms": round(latency_ms, 2) if latency_ms == latency_ms else "",
        "fps": round(fps, 2) if fps == fps else "",
        "param_count": speed["param_count"],
        "trainable_param_count": speed["trainable_param_count"],
    })
    return results


def print_results(model_kind: str, results: dict):
    print(f"\n{'=' * 60}")
    print(f"  {model_kind}  |  {results['num_images']} images")
    print(f"{'=' * 60}")
    print(f"  mAP@0.5       : {results['map50']:.4f}")
    print(f"  mAP@0.5:0.95  : {results['map50_95']:.4f}")
    print(f"  Precision     : {results['precision']:.4f}")
    print(f"  Recall        : {results['recall']:.4f}")
    print(f"  F1            : {results['f1']:.4f}")
    print(f"    (at IoU {results['pr_iou_threshold']}, conf {results['pr_confidence_threshold']})")
    print(f"  TP/FP/FN      : {results['tp']} / {results['fp']} / {results['fn']}")
    print("\n  Per-class AP@0.5:")
    for name in CLASS_NAMES:
        print(f"    {name}: {results['per_class_ap50'][name]:.4f}   "
              f"(GT boxes: {results['gt_counts'][name]})")
    print(f"\n  Parameters    : {results['param_count']:,}")
    if results["latency_ms"] != "":
        print(f"  Latency       : {results['latency_ms']} ms/image")
        print(f"  Throughput    : {results['fps']} FPS")
    print(f"{'=' * 60}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["yolo", "faster_rcnn"], required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--dataset", default="source", help="dataset name for the log row")
    ap.add_argument("--split", default="test", help="split name for the log row")
    ap.add_argument("--no-speed", action="store_true", help="skip latency measurement")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    hp = load_hyperparams()
    results = run_evaluation(
        args.model, Path(args.weights), Path(args.images), Path(args.labels),
        hp, measure_speed=not args.no_speed, device=args.device,
    )
    print_results(args.model, results)

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote {args.json_out}")

    return results


if __name__ == "__main__":
    main()
