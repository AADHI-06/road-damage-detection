"""
Visualization utilities: prediction-vs-ground-truth overlays and failure-example
selection, used by Phase 4's cross-country evaluation (CLAUDE.md section 7:
"Generate qualitative failure examples -- a few images per country where the
model fails, for the report's analysis section").

Also usable standalone for confusion-matrix-style eyeballing in later phases.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "eval"))

from metrics import box_iou, CLASS_NAMES  # noqa: E402

GT_COLOR = (0, 200, 0)     # green, BGR (cv2 convention)
PRED_COLOR = (0, 0, 255)   # red, BGR


def score_failure(pred: dict, gt: dict, iou_threshold: float = 0.5,
                  fp_conf_threshold: float = 0.5) -> dict:
    """How badly one image's predictions disagree with ground truth.

    Uses the SAME greedy highest-IoU-first matching rule as metrics.py's AP
    computation, per class, so "failure" here means the same thing "false
    positive/negative" means in the reported numbers -- not some separate,
    unaligned notion of error.

    Returns:
        {"fn": missed ground-truth boxes, "fp": confident wrong predictions
         (score >= fp_conf_threshold), "score": fn + fp}
    A higher score means a worse image to show as a qualitative failure case.
    Low-confidence false positives are excluded from the score (but not from
    drawing) so the ranking isn't dominated by many near-threshold noise boxes.
    """
    fn = 0
    fp = 0
    for class_id in range(len(CLASS_NAMES)):
        gt_mask = gt["labels"] == class_id
        pred_mask = pred["labels"] == class_id
        gt_boxes = gt["boxes"][gt_mask]
        pred_boxes = pred["boxes"][pred_mask]
        pred_scores = pred["scores"][pred_mask]

        if len(gt_boxes) == 0:
            fp += int((pred_scores >= fp_conf_threshold).sum())
            continue
        if len(pred_boxes) == 0:
            fn += len(gt_boxes)
            continue

        order = np.argsort(-pred_scores)
        pred_boxes_sorted = pred_boxes[order]
        pred_scores_sorted = pred_scores[order]
        ious = box_iou(pred_boxes_sorted, gt_boxes)
        claimed = np.zeros(len(gt_boxes), dtype=bool)

        for i, row in enumerate(ious):
            candidates = np.where((row >= iou_threshold) & (~claimed))[0]
            if len(candidates):
                best = candidates[np.argmax(row[candidates])]
                claimed[best] = True
            elif pred_scores_sorted[i] >= fp_conf_threshold:
                fp += 1

        fn += int((~claimed).sum())

    return {"fn": fn, "fp": fp, "score": fn + fp}


def select_failure_examples(image_paths: list[Path], preds: list[dict], gts: list[dict],
                            k: int = 5, **score_kwargs) -> list[dict]:
    """Rank images by score_failure() and return the top-k worst, worst first.

    Images with score 0 (perfect match) are excluded even if k would otherwise
    include them -- a "failure example" that isn't actually a failure is not
    useful for the report's analysis section.
    """
    scored = []
    for path, pred, gt in zip(image_paths, preds, gts):
        s = score_failure(pred, gt, **score_kwargs)
        if s["score"] > 0:
            scored.append({"image_path": path, "pred": pred, "gt": gt, **s})
    scored.sort(key=lambda d: -d["score"])
    return scored[:k]


def draw_gt_and_predictions(image_path: Path, gt: dict, pred: dict, out_path: Path,
                            conf_threshold: float = 0.25):
    """Render one image with GT boxes (green) and predictions above
    conf_threshold (red), save to out_path. Legend baked into the filename's
    sibling would be nicer, but a fixed color convention documented here (and
    repeated in the report caption) is simpler and matches the rest of the
    project's sanity-check scripts.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return False

    for box, cls_id in zip(gt["boxes"], gt["labels"]):
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(img, (x1, y1), (x2, y2), GT_COLOR, 2)
        cv2.putText(img, f"GT:{CLASS_NAMES[cls_id]}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, GT_COLOR, 1)

    h = img.shape[0]
    for box, cls_id, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
        if score < conf_threshold:
            continue
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(img, (x1, y1), (x2, y2), PRED_COLOR, 2)
        cv2.putText(img, f"{CLASS_NAMES[cls_id]}:{score:.2f}", (x1, min(h - 4, y2 + 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, PRED_COLOR, 1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return True
