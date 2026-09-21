"""
torch Dataset that reads the YOLO-format data built in Phase 2 and yields the
(image, target) pairs torchvision's Faster R-CNN expects.

Both detectors therefore train from the exact same files on disk -- there is no
second conversion path that could drift out of sync with the YOLO one.

TWO CONVERSIONS HAPPEN HERE, and both are classic silent-bug sources:

1. COORDINATES. YOLO stores normalized center-x, center-y, width, height in
   [0,1]. torchvision wants absolute xyxy (left, top, right, bottom) in pixels.
2. CLASS IDs. torchvision reserves label 0 for BACKGROUND, so the four damage
   classes must occupy labels 1..4, not 0..3. Everything is shifted by +1 on
   the way in and shifted back by -1 at evaluation time, so that metrics.py
   always sees the project's canonical 0..3 ids.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# Shift between project class ids (0..3) and torchvision's (1..4, 0 = background).
LABEL_OFFSET = 1

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


class YoloFormatDetectionDataset(Dataset):
    """Reads images/<split>/ + labels/<split>/ produced by src/data/build_splits.py.

    Args:
        images_dir: directory of images.
        labels_dir: directory of matching .txt YOLO labels (same stem).
        train: if True, apply horizontal-flip augmentation.
    """

    def __init__(self, images_dir: Path, labels_dir: Path, train: bool = False,
                 hflip_prob: float = 0.5, seed: int = 42):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.train = train
        self.hflip_prob = hflip_prob
        # Dedicated RNG so augmentation is reproducible and independent of any
        # other random draws happening elsewhere in the process.
        self.rng = np.random.default_rng(seed)

        self.image_paths = sorted(
            p for p in self.images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
        )
        if not self.image_paths:
            raise ValueError(f"No images found in {self.images_dir}")

    def __len__(self):
        return len(self.image_paths)

    def _load_boxes(self, image_path: Path, width: int, height: int):
        """Read one YOLO label file -> (boxes xyxy absolute, labels shifted by +1)."""
        label_path = self.labels_dir / f"{image_path.stem}.txt"
        boxes, labels = [], []

        if label_path.exists():
            text = label_path.read_text(encoding="utf-8").strip()
            for line in text.splitlines() if text else []:
                class_id, xc, yc, bw, bh = line.split()
                xc, yc, bw, bh = float(xc), float(yc), float(bw), float(bh)

                # Normalized center/size -> absolute corners.
                x1 = (xc - bw / 2) * width
                y1 = (yc - bh / 2) * height
                x2 = (xc + bw / 2) * width
                y2 = (yc + bh / 2) * height

                # Guard against degenerate boxes: torchvision's loss produces NaN
                # if a box has non-positive width or height.
                if x2 <= x1 or y2 <= y1:
                    continue

                boxes.append([x1, y1, x2, y2])
                labels.append(int(class_id) + LABEL_OFFSET)

        if boxes:
            return (torch.tensor(boxes, dtype=torch.float32),
                    torch.tensor(labels, dtype=torch.int64))
        # Background-only image: torchvision accepts these as negative samples,
        # but the tensors must still have the right shape and dtype.
        return (torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.int64))

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        img = Image.open(image_path).convert("RGB")
        width, height = img.size

        boxes, labels = self._load_boxes(image_path, width, height)

        if self.train and self.rng.random() < self.hflip_prob:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if len(boxes):
                # Mirror x coordinates; note left/right swap, so x1 and x2 exchange roles.
                x1 = boxes[:, 0].clone()
                x2 = boxes[:, 2].clone()
                boxes[:, 0] = width - x2
                boxes[:, 2] = width - x1

        # HWC uint8 -> CHW float in [0,1]. The model's own GeneralizedRCNNTransform
        # handles normalization and resizing after this point.
        img_tensor = torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1).float() / 255.0

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }
        return img_tensor, target


def collate_fn(batch):
    """Detection batches cannot be stacked: images differ in size and box count."""
    return tuple(zip(*batch))
