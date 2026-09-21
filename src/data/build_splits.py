"""
Phase 2: build the SOURCE and TARGET datasets used for the whole project
(CLAUDE.md section 5.1), converting VOC XML to YOLO txt as it goes.

SOURCE = India + Japan, split 70/15/15 at image level, seed=42.
         The split is done per-country then merged, so both train/val/test
         keep the same India:Japan ratio as the original data (a plain pooled
         shuffle could by chance skew one split toward one country).
TARGET = Czech, Norway, United_States, China_Drone, China_MotorBike -- each
         converted whole, no split, since these are only ever used for
         zero-shot evaluation. China's two subsets are kept SEPARATE rather
         than merged (CLAUDE.md section 4.3: "China appears as two subsets
         with different capture modalities... treat and report them
         separately, do not merge blindly" -- drone vs. vehicle-mounted is a
         genuinely different domain, and section 5.2 says never average away
         per-country variation, so merging them would hide exactly the effect
         Phase 4 exists to measure). See data/DATA_REPORT.md section 8.

Usage:
    python src/data/build_splits.py                # full rebuild (source + all targets)
    python src/data/build_splits.py --targets-only  # rebuild targets only, skip SOURCE

Writes:
    data/processed/source/images/{train,val,test}/*.jpg
    data/processed/source/labels/{train,val,test}/*.txt
    data/processed/target/<key>/images/*.jpg
    data/processed/target/<key>/labels/*.txt
    data/split_report.json   -- per-country/per-split counts and drop/clip stats
"""

import argparse
import json
import random
import shutil
from pathlib import Path

from voc_to_yolo import convert_annotation

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

SEED = 42
SOURCE_COUNTRIES = ["India", "Japan"]
TARGET_GROUPS = {
    "czech": ["Czech"],
    "norway": ["Norway"],
    "us": ["United_States"],
    "china_drone": ["China_Drone"],
    "china_motorbike": ["China_MotorBike"],
}
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def list_country_images(country: str) -> list[Path]:
    img_dir = DATA_DIR / country / "train" / "images"
    return sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def xml_path_for_image(country: str, image_path: Path) -> Path:
    return DATA_DIR / country / "train" / "annotations" / "xmls" / f"{image_path.stem}.xml"


def split_indices(n: int, seed: int) -> dict:
    """Deterministically shuffle 0..n-1 with `seed` and cut into 70/15/15."""
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_train = int(round(n * SPLIT_RATIOS["train"]))
    n_val = int(round(n * SPLIT_RATIOS["val"]))
    return {
        "train": idx[:n_train],
        "val": idx[n_train:n_train + n_val],
        "test": idx[n_train + n_val:],
    }


def new_stats() -> dict:
    return {"num_kept": 0, "num_dropped": 0, "num_clipped": 0, "dropped_classes": {}}


def merge_stats(into: dict, result: dict):
    into["num_kept"] += result["num_kept"]
    into["num_dropped"] += result["num_dropped"]
    into["num_clipped"] += result["num_clipped"]
    for cls, cnt in result["dropped_classes"].items():
        into["dropped_classes"][cls] = into["dropped_classes"].get(cls, 0) + cnt


def convert_and_write(image_path: Path, xml_path: Path, out_images_dir: Path,
                       out_labels_dir: Path, stats: dict):
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, out_images_dir / image_path.name)
    result = convert_annotation(xml_path)
    label_path = out_labels_dir / f"{image_path.stem}.txt"
    label_path.write_text("\n".join(result["lines"]), encoding="utf-8")
    merge_stats(stats, result)


def build_source_splits(report: dict):
    combined = {"train": [], "val": [], "test": []}
    report["source_per_country_counts"] = {}

    for country in SOURCE_COUNTRIES:
        images = list_country_images(country)
        splits = split_indices(len(images), SEED)
        for split_name, idxs in splits.items():
            combined[split_name] += [(country, images[i]) for i in idxs]
        report["source_per_country_counts"][country] = {
            "total": len(images),
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
        }

    stats = new_stats()
    report["source_split_sizes"] = {}
    for split_name, items in combined.items():
        out_images_dir = PROCESSED_DIR / "source" / "images" / split_name
        out_labels_dir = PROCESSED_DIR / "source" / "labels" / split_name
        for country, image_path in items:
            xml_path = xml_path_for_image(country, image_path)
            convert_and_write(image_path, xml_path, out_images_dir, out_labels_dir, stats)
        report["source_split_sizes"][split_name] = len(items)
        print(f"  source/{split_name}: {len(items)} images")

    report["source_conversion_stats"] = stats


def build_target_sets(report: dict):
    report["target_counts"] = {}
    report["target_conversion_stats"] = {}

    for key, countries in TARGET_GROUPS.items():
        stats = new_stats()
        out_images_dir = PROCESSED_DIR / "target" / key / "images"
        out_labels_dir = PROCESSED_DIR / "target" / key / "labels"
        count = 0
        for country in countries:
            images = list_country_images(country)
            count += len(images)
            for image_path in images:
                xml_path = xml_path_for_image(country, image_path)
                convert_and_write(image_path, xml_path, out_images_dir, out_labels_dir, stats)
        report["target_counts"][key] = count
        report["target_conversion_stats"][key] = stats
        print(f"  target/{key}: {count} images (from {countries})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets-only", action="store_true",
                    help="rebuild TARGET sets only; skip SOURCE (which is unchanged "
                         "and expensive to re-copy)")
    args = ap.parse_args()

    report_path = DATA_DIR / "split_report.json"
    # Preserve the existing source_* keys when only targets are being rebuilt,
    # so split_report.json still reflects the full picture rather than losing
    # the SOURCE section.
    report = {"seed": SEED, "split_ratios": SPLIT_RATIOS}
    if args.targets_only and report_path.exists():
        with open(report_path, encoding="utf-8") as f:
            existing = json.load(f)
        for key in ("source_per_country_counts", "source_split_sizes", "source_conversion_stats"):
            if key in existing:
                report[key] = existing[key]

    if not args.targets_only:
        print("Building SOURCE splits (India + Japan, 70/15/15, seed=42)...")
        build_source_splits(report)
    else:
        print("Skipping SOURCE (--targets-only).")

    print("Building TARGET zero-shot sets (Czech, Norway, US, China_Drone, China_MotorBike)...")
    build_target_sets(report)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
