"""
Build a small subset of the SOURCE data for end-to-end pipeline validation.

the project spec section 8: "start with YOLOv8n and a subset of the data to validate the
full pipeline end-to-end before launching long runs. A pipeline bug found after a
6-hour training run is the most expensive failure mode in this project."

This produces data/processed/source_subset/ with the same layout as the full
source set, so the identical training and evaluation code paths exercise it --
a smoke test that ran through different code would not prove much.

Sampling is stratified toward images that actually contain boxes: a uniform
random draw would be mostly empty images (see data/DATA_REPORT.md section 3),
and a smoke test where the model never sees a positive example cannot detect
a broken label pipeline.

Usage:
    python src/data/make_subset.py --train 300 --val 100 --test 100
"""

import argparse
import random
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "data" / "processed" / "source"
SUBSET_DIR = REPO_ROOT / "data" / "processed" / "source_subset"

SEED = 42
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
# Fraction of each subset split that must contain at least one box.
NONEMPTY_FRACTION = 0.7


def build_split(split: str, n: int, rng: random.Random):
    src_images = SOURCE_DIR / "images" / split
    src_labels = SOURCE_DIR / "labels" / split
    dst_images = SUBSET_DIR / "images" / split
    dst_labels = SUBSET_DIR / "labels" / split

    # Rebuild from scratch so repeated runs cannot accumulate stale files.
    for d in (dst_images, dst_labels):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in src_images.iterdir() if p.suffix.lower() in IMAGE_EXTS)

    nonempty, empty = [], []
    for image_path in images:
        label_path = src_labels / f"{image_path.stem}.txt"
        has_boxes = label_path.exists() and bool(label_path.read_text(encoding="utf-8").strip())
        (nonempty if has_boxes else empty).append(image_path)

    rng.shuffle(nonempty)
    rng.shuffle(empty)

    n_nonempty = min(len(nonempty), int(n * NONEMPTY_FRACTION))
    n_empty = min(len(empty), n - n_nonempty)
    chosen = nonempty[:n_nonempty] + empty[:n_empty]
    rng.shuffle(chosen)

    for image_path in chosen:
        shutil.copy2(image_path, dst_images / image_path.name)
        label_path = src_labels / f"{image_path.stem}.txt"
        shutil.copy2(label_path, dst_labels / label_path.name)

    print(f"  {split}: {len(chosen)} images ({n_nonempty} with boxes, {n_empty} empty)")
    return len(chosen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=300)
    ap.add_argument("--val", type=int, default=100)
    ap.add_argument("--test", type=int, default=100)
    args = ap.parse_args()

    rng = random.Random(SEED)
    print(f"Building subset at {SUBSET_DIR}")
    for split, n in (("train", args.train), ("val", args.val), ("test", args.test)):
        build_split(split, n, rng)
    print("Done.")


if __name__ == "__main__":
    main()
