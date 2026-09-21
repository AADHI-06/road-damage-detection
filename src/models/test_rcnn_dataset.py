"""
Correctness checks for the YOLO -> torchvision conversion in rcnn_dataset.py.

The two conversions it performs (normalized cxcywh -> absolute xyxy, and the
+1 class-id shift for torchvision's background label) are silent-failure
prone: a mistake in either still trains, still converges to something, and
just produces quietly wrong numbers. These tests pin both down against
hand-computed values.

Run:
    python src/models/test_rcnn_dataset.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rcnn_dataset import YoloFormatDetectionDataset, LABEL_OFFSET  # noqa: E402


def build_fixture(tmp: Path):
    """One 200x100 image with one box: class 0, centered, half width/height."""
    images_dir = tmp / "images"
    labels_dir = tmp / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    Image.fromarray(np.zeros((100, 200, 3), dtype=np.uint8)).save(images_dir / "a.jpg")
    # cx=0.5, cy=0.5, w=0.5, h=0.5 on a 200x100 image
    #   -> x1 = (0.5-0.25)*200 = 50, y1 = (0.5-0.25)*100 = 25
    #   -> x2 = (0.5+0.25)*200 = 150, y2 = (0.5+0.25)*100 = 75
    (labels_dir / "a.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    return images_dir, labels_dir


def test_coordinate_conversion():
    with tempfile.TemporaryDirectory() as td:
        images_dir, labels_dir = build_fixture(Path(td))
        ds = YoloFormatDetectionDataset(images_dir, labels_dir, train=False)
        _, target = ds[0]

        boxes = target["boxes"].numpy()
        assert boxes.shape == (1, 4), boxes.shape
        np.testing.assert_allclose(boxes[0], [50.0, 25.0, 150.0, 75.0], atol=1e-4)
        print("  normalized cxcywh -> absolute xyxy: OK")


def test_label_offset():
    with tempfile.TemporaryDirectory() as td:
        images_dir, labels_dir = build_fixture(Path(td))
        ds = YoloFormatDetectionDataset(images_dir, labels_dir, train=False)
        _, target = ds[0]

        # Project class 0 (D00) must arrive as torchvision label 1, because
        # torchvision reserves 0 for background.
        assert int(target["labels"][0]) == 0 + LABEL_OFFSET, target["labels"]
        assert LABEL_OFFSET == 1
        print("  class id shifted by +1 for background: OK")


def test_empty_label_file():
    """Background-only images must yield correctly shaped empty tensors, not crash."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        images_dir, labels_dir = build_fixture(tmp)
        Image.fromarray(np.zeros((100, 200, 3), dtype=np.uint8)).save(images_dir / "b.jpg")
        (labels_dir / "b.txt").write_text("", encoding="utf-8")

        ds = YoloFormatDetectionDataset(images_dir, labels_dir, train=False)
        # Files are sorted, so index 1 is "b".
        _, target = ds[1]
        assert target["boxes"].shape == (0, 4), target["boxes"].shape
        assert target["labels"].shape == (0,), target["labels"].shape
        assert target["boxes"].dtype.is_floating_point
        print("  empty label file -> shaped empty tensors: OK")


def test_horizontal_flip_mirrors_boxes():
    """Flip augmentation must mirror x coords and keep x1 < x2."""
    with tempfile.TemporaryDirectory() as td:
        images_dir, labels_dir = build_fixture(Path(td))
        # hflip_prob=1.0 forces the flip so the result is deterministic.
        ds = YoloFormatDetectionDataset(images_dir, labels_dir, train=True, hflip_prob=1.0)
        _, target = ds[0]

        boxes = target["boxes"].numpy()[0]
        # Original x span was [50, 150] on width 200; mirrored is [200-150, 200-50] = [50, 150].
        # That box is symmetric, so also check ordering invariants hold.
        assert boxes[0] < boxes[2], boxes
        assert boxes[1] < boxes[3], boxes
        np.testing.assert_allclose(boxes, [50.0, 25.0, 150.0, 75.0], atol=1e-4)
        print("  horizontal flip mirrors boxes correctly: OK")


def test_asymmetric_box_flip():
    """A box not centered horizontally must actually move when flipped."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        images_dir = tmp / "images"
        labels_dir = tmp / "labels"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        Image.fromarray(np.zeros((100, 200, 3), dtype=np.uint8)).save(images_dir / "a.jpg")
        # cx=0.25 -> x spans [ (0.25-0.05)*200, (0.25+0.05)*200 ] = [40, 60]
        (labels_dir / "a.txt").write_text("2 0.25 0.5 0.1 0.5\n", encoding="utf-8")

        ds = YoloFormatDetectionDataset(images_dir, labels_dir, train=True, hflip_prob=1.0)
        _, target = ds[0]
        boxes = target["boxes"].numpy()[0]
        # Mirrored: [200-60, 200-40] = [140, 160]
        np.testing.assert_allclose(boxes[[0, 2]], [140.0, 160.0], atol=1e-4)
        assert int(target["labels"][0]) == 2 + LABEL_OFFSET
        print("  off-center box mirrors to correct position: OK")


if __name__ == "__main__":
    print("Running rcnn_dataset conversion tests...")
    test_coordinate_conversion()
    test_label_offset()
    test_empty_label_file()
    test_horizontal_flip_mirrors_boxes()
    test_asymmetric_box_flip()
    print("\nAll rcnn_dataset tests passed.")
