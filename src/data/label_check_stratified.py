"""
Stratified Phase 2 sanity check: 20 converted SOURCE (India + Japan) samples,
with minimums enforced so the rarest class (D40 / pothole) and both source
countries actually show up in what gets eyeballed, rather than trusting a
plain random draw to include them. Complements the plain random sample in
src/data/sanity_check_labels.py.

Sampling order (seed=42, deterministic):
  1. Fill the D40 quota first (pothole boxes are rare -- see data/DATA_REPORT.md
     section 3 -- so drawing them last risks missing them entirely).
  2. Top up India and Japan to their per-country minimums.
  3. Fill any remaining slots uniformly at random from what's left.

Reads data/processed/source/ only. Writes annotated images only -- no
training, no model code involved.

Usage:
    python src/data/label_check_stratified.py

Writes:
    experiments/results/figures/label_check/*.jpg
Prints:
    Summary table of sampled files (country, split, box count, classes present).
"""

import random
from pathlib import Path

import cv2

from voc_to_yolo import CLASS_TO_ID

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "data" / "processed" / "source"
OUT_DIR = REPO_ROOT / "experiments" / "results" / "figures" / "label_check"

ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}
CLASS_COLORS = {  # BGR
    "D00": (255, 0, 0),
    "D10": (0, 255, 0),
    "D20": (0, 165, 255),
    "D40": (0, 0, 255),
}

SEED = 42
TOTAL_SAMPLES = 20
MIN_INDIA = 5
MIN_JAPAN = 5
MIN_D40 = 3
D40_CLASS_ID = CLASS_TO_ID["D40"]


def country_of(image_path: Path) -> str:
    stem = image_path.stem
    if stem.startswith("India_"):
        return "India"
    if stem.startswith("Japan_"):
        return "Japan"
    raise ValueError(f"data/processed/source/ should only contain India_/Japan_ files, got: {image_path.name}")


def load_boxes(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    boxes = []
    for line in text.splitlines():
        class_id, xc, yc, w, h = line.split()
        boxes.append((int(class_id), float(xc), float(yc), float(w), float(h)))
    return boxes


def collect_pool() -> list[dict]:
    """Every SOURCE (image, label) pair across all splits, with country/box info attached."""
    pool = []
    images_root = SOURCE_DIR / "images"
    labels_root = SOURCE_DIR / "labels"
    for split_dir in sorted(images_root.iterdir()):
        split = split_dir.name
        for image_path in sorted(split_dir.iterdir()):
            label_path = labels_root / split / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            boxes = load_boxes(label_path)
            pool.append({
                "image_path": image_path,
                "label_path": label_path,
                "split": split,
                "country": country_of(image_path),
                "boxes": boxes,
                "has_d40": any(b[0] == D40_CLASS_ID for b in boxes),
            })
    return pool


def stratified_sample(pool: list[dict], rng: random.Random) -> list[dict]:
    selected = []
    selected_paths = set()

    def add(item):
        if item["image_path"] not in selected_paths:
            selected.append(item)
            selected_paths.add(item["image_path"])

    d40_pool = [x for x in pool if x["has_d40"]]
    rng.shuffle(d40_pool)
    for item in d40_pool[:MIN_D40]:
        add(item)

    india_pool = [x for x in pool if x["country"] == "India" and x["image_path"] not in selected_paths]
    rng.shuffle(india_pool)
    india_have = sum(1 for x in selected if x["country"] == "India")
    for item in india_pool[:max(0, MIN_INDIA - india_have)]:
        add(item)

    japan_pool = [x for x in pool if x["country"] == "Japan" and x["image_path"] not in selected_paths]
    rng.shuffle(japan_pool)
    japan_have = sum(1 for x in selected if x["country"] == "Japan")
    for item in japan_pool[:max(0, MIN_JAPAN - japan_have)]:
        add(item)

    remaining_pool = [x for x in pool if x["image_path"] not in selected_paths]
    rng.shuffle(remaining_pool)
    for item in remaining_pool:
        if len(selected) >= TOTAL_SAMPLES:
            break
        add(item)

    return selected[:TOTAL_SAMPLES]


def draw_and_save(item: dict) -> Path:
    img = cv2.imread(str(item["image_path"]))
    h, w = img.shape[:2]
    for class_id, xc, yc, bw, bh in item["boxes"]:
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        cls_name = ID_TO_CLASS[class_id]
        color = CLASS_COLORS[cls_name]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, cls_name, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    out_path = OUT_DIR / item["image_path"].name
    cv2.imwrite(str(out_path), img)
    return out_path


def main():
    rng = random.Random(SEED)
    pool = collect_pool()
    sample = stratified_sample(pool, rng)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for item in sample:
        out_path = draw_and_save(item)
        classes_present = sorted({ID_TO_CLASS[c] for c, *_ in item["boxes"]})
        rows.append({
            "file": item["image_path"].name,
            "country": item["country"],
            "split": item["split"],
            "num_boxes": len(item["boxes"]),
            "classes": ",".join(classes_present) if classes_present else "(none)",
        })

    india_count = sum(1 for r in rows if r["country"] == "India")
    japan_count = sum(1 for r in rows if r["country"] == "Japan")
    d40_count = sum(1 for r in rows if "D40" in r["classes"])

    print(f"Sampled {len(rows)} images  (India={india_count}, Japan={japan_count}, containing D40={d40_count})\n")
    header = f"{'File':<22} {'Country':<8} {'Split':<6} {'Boxes':>5}  Classes"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['file']:<22} {r['country']:<8} {r['split']:<6} {r['num_boxes']:>5}  {r['classes']}")

    print(f"\nWrote {len(rows)} annotated images to {OUT_DIR}")


if __name__ == "__main__":
    main()
