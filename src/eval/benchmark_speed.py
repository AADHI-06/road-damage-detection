"""
Inference latency, throughput (FPS), and parameter count for both detectors
(CLAUDE.md section 5.3, and section 7 Phase 3a: "Benchmark inference speed and
parameter count for both").

These three numbers are what back the report's accuracy-vs-speed trade-off
discussion. Two rules make them meaningful:

1. BOTH MODELS MUST BE MEASURED ON THE SAME HARDWARE. Latency in ms is
   meaningless across machines, so experiment_log.csv records a hardware string
   on every row and comparisons are only valid within one.
2. LATENCY IS MEASURED AT BATCH SIZE 1. Batching raises throughput but hides
   per-image response time, which is what matters for a deployed road survey.

Parameter count is architecture-only and hardware-independent, so it is the one
figure here that stays comparable across machines.

Usage:
    # both models, identical conditions
    python src/eval/benchmark_speed.py \
        --yolo-weights experiments/runs/yolov8n_source/weights/best.pt \
        --rcnn-weights experiments/runs/faster_rcnn_source/best.pt \
        --images data/processed/source/images/test --device cpu
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "models"))

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
LATENCY_SAMPLE_SIZE = 100  # images timed one at a time
LATENCY_WARMUP = 5         # discarded: first passes include lazy init / cache warming
NUM_CLASSES_WITH_BACKGROUND = 5  # 4 damage classes + background


def load_hyperparams() -> dict:
    with open(REPO_ROOT / "config" / "hyperparams.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_images(images_dir: Path) -> list[Path]:
    return sorted(p for p in Path(images_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS)


def count_parameters(model) -> dict:
    """Total and trainable parameter counts for an already-built torch module.

    `trainable` is often 0 for a YOLO checkpoint: ultralytics strips the
    optimizer and freezes parameters when exporting best.pt for inference.
    That is expected for an inference checkpoint, not a bug -- report
    `param_count` (the architecture size) in comparison tables.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"param_count": int(total), "trainable_param_count": int(trainable)}


def load_model(model_kind: str, weights: Path, imgsz: int, device: str):
    """Return (callable_taking_one_image_path, underlying_torch_module)."""
    if model_kind == "yolo":
        from ultralytics import YOLO

        yolo = YOLO(str(weights))

        def run(image_path):
            yolo.predict([str(image_path)], imgsz=imgsz, verbose=False, device=device)

        # .model is the underlying torch module; YOLO itself is a wrapper.
        return run, yolo.model

    import torch
    from PIL import Image
    from rcnn_model import build_faster_rcnn

    model = build_faster_rcnn(NUM_CLASSES_WITH_BACKGROUND, imgsz, pretrained=False,
                              checkpoint=weights)
    model.eval()
    torch_device = torch.device(device)
    model.to(torch_device)

    def run(image_path):
        img = Image.open(image_path).convert("RGB")
        tensor = torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1).float() / 255.0
        with torch.no_grad():
            model([tensor.to(torch_device)])

    return run, model


def measure_latency(model_kind: str, weights: Path, image_paths: list[Path],
                    imgsz: int, device: str = "cpu") -> dict:
    """Time single-image inference and count parameters.

    Returns latency_ms (mean), latency_ms_median, fps, and parameter counts.
    The median is reported alongside the mean because a single OS hiccup can
    skew the mean on a laptop; a large mean/median gap means the measurement
    was disturbed and should be repeated.
    """
    import torch

    run, module = load_model(model_kind, weights, imgsz, device)
    params = count_parameters(module)

    sample = image_paths[:LATENCY_SAMPLE_SIZE + LATENCY_WARMUP]
    if not sample:
        return {**params, "latency_ms": float("nan"), "fps": float("nan")}

    timings = []
    for i, path in enumerate(sample):
        t0 = time.perf_counter()
        run(path)
        # CUDA kernels are asynchronous: without a sync the timer would measure
        # queue submission, not actual compute.
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if i >= LATENCY_WARMUP:
            timings.append(dt)

    if not timings:
        return {**params, "latency_ms": float("nan"), "fps": float("nan")}

    mean_s = float(np.mean(timings))
    return {
        **params,
        "latency_ms": mean_s * 1000.0,
        "latency_ms_median": float(np.median(timings)) * 1000.0,
        "fps": 1.0 / mean_s,
        "num_timed_images": len(timings),
    }


def print_benchmark(label: str, r: dict):
    print(f"\n{label}")
    print(f"  parameters      : {r['param_count']:,}  ({r['trainable_param_count']:,} trainable)")
    if r["latency_ms"] == r["latency_ms"]:  # not NaN
        print(f"  latency (mean)  : {r['latency_ms']:.2f} ms/image")
        print(f"  latency (median): {r.get('latency_ms_median', float('nan')):.2f} ms/image")
        print(f"  throughput      : {r['fps']:.2f} FPS")
        print(f"  timed on        : {r.get('num_timed_images', 0)} images (batch size 1)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo-weights", default=None)
    ap.add_argument("--rcnn-weights", default=None)
    ap.add_argument("--images", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if not args.yolo_weights and not args.rcnn_weights:
        raise SystemExit("Pass --yolo-weights and/or --rcnn-weights")

    hp = load_hyperparams()
    image_paths = list_images(Path(args.images))
    print(f"Benchmarking on {len(image_paths)} images, device={args.device}")

    if args.yolo_weights:
        r = measure_latency("yolo", Path(args.yolo_weights), image_paths,
                            hp["yolo"]["imgsz"], args.device)
        print_benchmark("YOLOv8 (plain)", r)

    if args.rcnn_weights:
        r = measure_latency("faster_rcnn", Path(args.rcnn_weights), image_paths,
                            hp["faster_rcnn"]["imgsz"], args.device)
        print_benchmark("Faster R-CNN (plain)", r)

    print("\nNote: latency/FPS are only comparable between models measured on the "
          "same hardware. Parameter count is hardware-independent.")


if __name__ == "__main__":
    main()
