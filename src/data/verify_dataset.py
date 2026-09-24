"""
Phase 1 script: walk the actual downloaded RDD2022 directory and report what is
really there. Do NOT assume the layout documented in the project spec is correct --
this script's job is to catch cases where it isn't (missing splits, orphaned
files, unparsable XML) before any conversion or training code depends on them.

Usage:
    python src/data/verify_dataset.py

Writes:
    data/verification_report.json   (machine-readable, consumed by dataset_stats.py)
Prints:
    Human-readable summary to stdout.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# Project root = two levels up from this file (src/data/verify_dataset.py -> repo root)
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"


def list_files_by_stem(directory: Path, exts: set[str]) -> dict[str, Path]:
    """Map filename-without-extension -> path, for files matching exts (case-insensitive)."""
    if not directory.exists():
        return {}
    out = {}
    for p in directory.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            out[p.stem] = p
    return out


def check_xml_parseable(xml_path: Path) -> tuple[bool, str | None]:
    try:
        ET.parse(xml_path)
        return True, None
    except ET.ParseError as e:
        return False, str(e)


def verify_country(country_dir: Path) -> dict:
    country = country_dir.name
    result = {"country": country}

    train_images_dir = country_dir / "train" / "images"
    train_xml_dir = country_dir / "train" / "annotations" / "xmls"
    test_images_dir = country_dir / "test" / "images"

    train_images = list_files_by_stem(train_images_dir, IMAGE_EXTS)
    train_xmls = list_files_by_stem(train_xml_dir, {".xml"})
    test_images = list_files_by_stem(test_images_dir, IMAGE_EXTS)

    result["train_images_dir_exists"] = train_images_dir.exists()
    result["train_xml_dir_exists"] = train_xml_dir.exists()
    result["test_images_dir_exists"] = test_images_dir.exists()

    result["num_train_images"] = len(train_images)
    result["num_train_xmls"] = len(train_xmls)
    result["num_test_images"] = len(test_images)

    # Orphans: images with no matching annotation XML, and XML with no matching image.
    # Matching is done by filename stem (e.g. "India_000010").
    image_stems = set(train_images)
    xml_stems = set(train_xmls)
    orphan_images = sorted(image_stems - xml_stems)
    orphan_xmls = sorted(xml_stems - image_stems)

    result["num_orphan_images_no_xml"] = len(orphan_images)
    result["num_orphan_xmls_no_image"] = len(orphan_xmls)
    result["orphan_images_sample"] = orphan_images[:10]
    result["orphan_xmls_sample"] = orphan_xmls[:10]

    # XML parse failures + cross-check the <filename> tag inside each XML against
    # the actual image directory (catches cases where the XML exists, parses fine,
    # but points at a filename that isn't present under this exact name/case).
    parse_failures = []
    filename_mismatches = []
    for stem, xml_path in train_xmls.items():
        ok, err = check_xml_parseable(xml_path)
        if not ok:
            parse_failures.append({"file": xml_path.name, "error": err})
            continue
        tree = ET.parse(xml_path)
        root = tree.getroot()
        filename_tag = root.findtext("filename")
        if filename_tag and filename_tag not in {p.name for p in train_images.values()}:
            filename_mismatches.append({"xml": xml_path.name, "xml_filename_tag": filename_tag})

    result["num_xml_parse_failures"] = len(parse_failures)
    result["xml_parse_failures"] = parse_failures
    result["num_xml_filename_mismatches"] = len(filename_mismatches)
    result["xml_filename_mismatches_sample"] = filename_mismatches[:10]

    return result


def main():
    if not DATA_DIR.exists():
        raise SystemExit(f"Data directory not found: {DATA_DIR}")

    country_dirs = sorted(p for p in DATA_DIR.iterdir() if p.is_dir())
    report = {"data_dir": str(DATA_DIR), "countries": []}

    print(f"Scanning {DATA_DIR}")
    print(f"Found {len(country_dirs)} country folders: {[p.name for p in country_dirs]}\n")

    for country_dir in country_dirs:
        r = verify_country(country_dir)
        report["countries"].append(r)

        print(f"=== {r['country']} ===")
        print(f"  train images        : {r['num_train_images']}")
        print(f"  train xml annotations: {r['num_train_xmls']}")
        print(f"  test images (unlabeled): {r['num_test_images']}")
        print(f"  orphan images (no xml): {r['num_orphan_images_no_xml']}")
        print(f"  orphan xmls (no image): {r['num_orphan_xmls_no_image']}")
        print(f"  xml parse failures    : {r['num_xml_parse_failures']}")
        print(f"  xml filename mismatches: {r['num_xml_filename_mismatches']}")
        print()

    out_path = DATA_DIR / "verification_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
