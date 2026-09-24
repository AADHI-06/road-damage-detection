"""
Phase 2: convert RDD2022 Pascal VOC XML annotations to YOLO normalized txt format.

YOLO format (one line per box): "<class_id> <x_center> <y_center> <width> <height>"
all four values normalized to [0, 1] by image width/height.

Only the four CRDDC-2022 challenge classes are kept (the project spec section 4.2); any other
code is dropped and counted so the drop total can be reported, not silently lost.

This module is imported by build_splits.py; it is not meant to be run standalone,
since "which images go where" is decided by the split logic, not this file.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

# Fixed class -> id mapping used everywhere in this project (train, eval, ablation).
CLASS_TO_ID = {"D00": 0, "D10": 1, "D20": 2, "D40": 3}


def convert_annotation(xml_path: Path) -> dict:
    """Parse one VOC XML file and return YOLO-format lines plus conversion stats.

    Returns:
        {
          "lines": ["<class_id> <xc> <yc> <w> <h>", ...],
          "num_kept": int,
          "num_dropped": int,       # boxes whose class is not one of the 4 target classes
          "dropped_classes": {code: count},
          "num_clipped": int,       # boxes whose box coordinates fell outside the image
                                     # and had to be clamped to [0, width]/[0, height]
                                     # before normalizing (protects against boxes that
                                     # would otherwise normalize outside [0, 1])
        }
    """
    root = ET.parse(xml_path).getroot()
    size_tag = root.find("size")
    width = int(size_tag.findtext("width"))
    height = int(size_tag.findtext("height"))

    lines = []
    num_dropped = 0
    dropped_classes = {}
    num_clipped = 0

    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name not in CLASS_TO_ID:
            num_dropped += 1
            dropped_classes[name] = dropped_classes.get(name, 0) + 1
            continue

        bnd = obj.find("bndbox")
        xmin = float(bnd.findtext("xmin"))
        ymin = float(bnd.findtext("ymin"))
        xmax = float(bnd.findtext("xmax"))
        ymax = float(bnd.findtext("ymax"))

        clipped_xmin = min(max(xmin, 0), width)
        clipped_ymin = min(max(ymin, 0), height)
        clipped_xmax = min(max(xmax, 0), width)
        clipped_ymax = min(max(ymax, 0), height)
        if (clipped_xmin, clipped_ymin, clipped_xmax, clipped_ymax) != (xmin, ymin, xmax, ymax):
            num_clipped += 1
        xmin, ymin, xmax, ymax = clipped_xmin, clipped_ymin, clipped_xmax, clipped_ymax

        x_center = (xmin + xmax) / 2 / width
        y_center = (ymin + ymax) / 2 / height
        box_w = (xmax - xmin) / width
        box_h = (ymax - ymin) / height

        # Defensive check, not expected to trigger given the clamping above -- if it
        # ever does, that means the XML's own <size> tag disagrees with the box coords
        # badly enough that clamping wasn't enough (e.g. size tag wrong for this image).
        assert 0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0, f"{xml_path}: center out of [0,1]"
        assert 0.0 <= box_w <= 1.0 and 0.0 <= box_h <= 1.0, f"{xml_path}: box size out of [0,1]"

        class_id = CLASS_TO_ID[name]
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

    return {
        "lines": lines,
        "num_kept": len(lines),
        "num_dropped": num_dropped,
        "dropped_classes": dropped_classes,
        "num_clipped": num_clipped,
    }
