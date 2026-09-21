"""
Phase 2 sanity check (CLAUDE.md section 7, Phase 2): render ~20 randomly sampled
converted YOLO labels back onto their source images. This is the step that catches
VOC->YOLO conversion bugs (flipped x/y, wrong normalization, off-by-one class ids)
that would otherwise silently corrupt every downstream training run. Do not skip.

Must be run AFTER build_splits.py (reads its output under data/processed/).

Usage:
    python src/data/sanity_check_labels.py

Writes:
    data/processed/sanity_check/*.jpg   -- images with boxes + class labels drawn on
"""

import random
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
OUT_DIR = PROCESSED_DIR / "sanity_check"

SEED = 42
NUM_SAMPLES = 20

ID_TO_CLASS = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}
CLASS_COLORS = {  # BGR
    "D00": (255, 0, 0),
    "D10": (0, 255, 0),
    "D20": (0, 165, 255),
    "D40": (0, 0, 255),
}


def find_image_label_pairs() -> list[tuple[Path, Path]]:
    """All (image_path, label_path) pairs across every source split and target group."""
    pairs = []

    source_images_root = PROCESSED_DIR / "source" / "images"
    if source_images_root.exists():
        for split_dir in source_images_root.iterdir():
            labels_dir = PROCESSED_DIR / "source" / "labels" / split_dir.name
            for img_path in split_dir.iterdir():
                label_path = labels_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    pairs.append((img_path, label_path))

    target_root = PROCESSED_DIR / "target"
    if target_root.exists():
        for group_dir in target_root.iterdir():
            images_dir = group_dir / "images"
            labels_dir = group_dir / "labels"
            if not images_dir.exists():
                continue
            for img_path in images_dir.iterdir():
                label_path = labels_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    pairs.append((img_path, label_path))

    return pairs


def draw_labels(image_path: Path, label_path: Path, out_path: Path):
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]

    text = label_path.read_text(encoding="utf-8").strip()
    lines = text.splitlines() if text else []

    for line in lines:
        parts = line.split()
        class_id = int(parts[0])
        xc, yc, bw, bh = (float(v) for v in parts[1:5])

        # De-normalize: YOLO stores center-x/y and width/height as fractions of image size.
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        cls_name = ID_TO_CLASS[class_id]
        color = CLASS_COLORS[cls_name]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, cls_name, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imwrite(str(out_path), img)


def main():
    pairs = find_image_label_pairs()
    if not pairs:
        raise SystemExit("No converted image/label pairs found. Run build_splits.py first.")

    sample = random.Random(SEED).sample(pairs, k=min(NUM_SAMPLES, len(pairs)))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for image_path, label_path in sample:
        out_path = OUT_DIR / image_path.name
        draw_labels(image_path, label_path, out_path)
        print(f"  {image_path.name} -> {out_path}")

    print(f"\nWrote {len(sample)} annotated sample images to {OUT_DIR}")
    print("Manually inspect these to confirm boxes align with visible damage before trusting the pipeline.")


if __name__ == "__main__":
    main()
