"""
Detection metrics: mAP@0.5, mAP@0.5:0.95, per-class AP, and Precision/Recall/F1.

WHY THIS EXISTS INSTEAD OF USING EACH FRAMEWORK'S BUILT-IN METRICS:
Ultralytics reports its own mAP for YOLOv8, and torchvision has no built-in
detection metric at all. If YOLOv8 were scored by ultralytics and Faster R-CNN
by something else, any measured gap between the two models would be partly an
artifact of two different metric implementations (they differ in interpolation
scheme, confidence flooring, and max-detection caps). Since the whole project
is a model x country comparison, both models are instead run through THIS
single implementation on identical inputs. The comparison is then apples-to-apples.

DEFINITIONS USED (state these in the report -- they are likely viva questions):
- A prediction is a True Positive if it has IoU >= t with a ground-truth box of
  the SAME class that no higher-scoring prediction has already claimed.
  Each ground-truth box can be matched at most once; extra detections of the
  same object are False Positives.
- AP is the area under the precision-recall curve, computed with all-point
  interpolation (VOC2010+ style): precision is made monotonically non-increasing
  before integrating, which removes the "sawtooth" of the raw curve.
- mAP@0.5 is AP at IoU 0.50 averaged over classes.
- mAP@0.5:0.95 averages AP over IoU thresholds 0.50, 0.55, ..., 0.95, then over
  classes -- the COCO convention.
- Precision/Recall/F1 are single-threshold metrics (unlike AP, which sweeps
  confidence). They are computed at a fixed confidence AND IoU threshold, and
  micro-averaged across classes (pool TP/FP/FN over all classes, then divide),
  which is the "overall F1" CRDDC-2022 ranked teams on.
"""

import numpy as np

CLASS_NAMES = ["D00", "D10", "D20", "D40"]


def box_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """IoU between every box in A and every box in B.

    Args:
        boxes_a: (N, 4) in xyxy absolute pixel coords.
        boxes_b: (M, 4) in xyxy absolute pixel coords.
    Returns:
        (N, M) IoU matrix.
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)

    # Broadcast A over rows and B over columns to get all pairwise intersections.
    lt = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])  # top-left
    rb = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])  # bottom-right
    wh = np.clip(rb - lt, a_min=0, a_max=None)
    inter = wh[..., 0] * wh[..., 1]

    area_a = np.clip(boxes_a[:, 2] - boxes_a[:, 0], 0, None) * np.clip(boxes_a[:, 3] - boxes_a[:, 1], 0, None)
    area_b = np.clip(boxes_b[:, 2] - boxes_b[:, 0], 0, None) * np.clip(boxes_b[:, 3] - boxes_b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter

    # np.where guards against zero-area boxes producing divide-by-zero warnings.
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def _average_precision(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Area under the PR curve using all-point interpolation.

    The raw precision-recall curve is jagged: precision jumps up every time a
    true positive lands. Interpolation replaces each precision value with the
    maximum precision achieved at any equal-or-higher recall, which makes the
    curve monotonically non-increasing before integrating.
    """
    # Sentinel endpoints so the curve spans recall 0 -> 1.
    mrec = np.concatenate([[0.0], recalls, [1.0]])
    mpre = np.concatenate([[0.0], precisions, [0.0]])

    # Sweep right-to-left, carrying the running maximum precision backwards.
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Integrate only where recall actually changes (each step contributes width * height).
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _ap_for_class(preds, gts, class_id: int, iou_threshold: float) -> tuple[float, int]:
    """AP for one class at one IoU threshold.

    Args:
        preds: list (per image) of dicts with keys boxes (N,4), scores (N,), labels (N,)
        gts:   list (per image) of dicts with keys boxes (M,4), labels (M,)
    Returns:
        (ap, num_ground_truths). AP is 0.0 when the class has no ground truth
        anywhere -- callers should exclude those classes from the mean.
    """
    # Collect every prediction of this class, tagged with its image index.
    image_indices, scores, boxes = [], [], []
    for img_idx, p in enumerate(preds):
        mask = p["labels"] == class_id
        if mask.any():
            image_indices.append(np.full(mask.sum(), img_idx))
            scores.append(p["scores"][mask])
            boxes.append(p["boxes"][mask])

    gt_boxes_by_image = {}
    num_gt = 0
    for img_idx, g in enumerate(gts):
        mask = g["labels"] == class_id
        n = int(mask.sum())
        if n:
            gt_boxes_by_image[img_idx] = g["boxes"][mask]
            num_gt += n

    if num_gt == 0:
        return 0.0, 0
    if not scores:
        return 0.0, num_gt

    image_indices = np.concatenate(image_indices)
    scores = np.concatenate(scores)
    boxes = np.concatenate(boxes)

    # Rank all predictions by confidence: the PR curve is traced by walking this order.
    order = np.argsort(-scores)
    image_indices, boxes = image_indices[order], boxes[order]

    # Track which ground-truth boxes have already been claimed, per image.
    matched = {img_idx: np.zeros(len(b), dtype=bool) for img_idx, b in gt_boxes_by_image.items()}

    tp = np.zeros(len(boxes))
    fp = np.zeros(len(boxes))

    for i, (img_idx, box) in enumerate(zip(image_indices, boxes)):
        gt_boxes = gt_boxes_by_image.get(int(img_idx))
        if gt_boxes is None:
            fp[i] = 1  # prediction in an image with no GT of this class
            continue

        ious = box_iou(box[None, :], gt_boxes)[0]
        best = int(np.argmax(ious))
        # Claim the best-overlapping GT, but only if it clears the threshold
        # and is still unclaimed. Otherwise this is a duplicate/false detection.
        if ious[best] >= iou_threshold and not matched[int(img_idx)][best]:
            tp[i] = 1
            matched[int(img_idx)][best] = True
        else:
            fp[i] = 1

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls = cum_tp / num_gt
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)

    return _average_precision(recalls, precisions), num_gt


def _micro_prf1(preds, gts, num_classes: int, iou_threshold: float,
                confidence_threshold: float) -> dict:
    """Precision / Recall / F1 pooled over all classes at fixed IoU + confidence.

    Micro-averaging (pool TP/FP/FN first, then divide) weights each detection
    equally, so common classes dominate. That is what CRDDC-2022's overall F1
    does. Macro-averaging would instead weight each class equally -- worth
    noting in the report given how rare D40 is.
    """
    total_tp = total_fp = total_fn = 0

    for p, g in zip(preds, gts):
        keep = p["scores"] >= confidence_threshold
        pred_boxes, pred_labels, pred_scores = p["boxes"][keep], p["labels"][keep], p["scores"][keep]

        for class_id in range(num_classes):
            pm = pred_labels == class_id
            gm = g["labels"] == class_id
            pb, ps = pred_boxes[pm], pred_scores[pm]
            gb = g["boxes"][gm]

            if len(gb) == 0:
                total_fp += len(pb)
                continue
            if len(pb) == 0:
                total_fn += len(gb)
                continue

            # Greedy matching in descending confidence, same rule as the AP path.
            order = np.argsort(-ps)
            pb = pb[order]
            ious = box_iou(pb, gb)
            claimed = np.zeros(len(gb), dtype=bool)

            for row in ious:
                candidate = np.where((row >= iou_threshold) & (~claimed))[0]
                if len(candidate):
                    best = candidate[np.argmax(row[candidate])]
                    claimed[best] = True
                    total_tp += 1
                else:
                    total_fp += 1

            total_fn += int((~claimed).sum())

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }


def confusion_matrix(preds, gts, num_classes: int, iou_threshold: float = 0.50,
                     confidence_threshold: float = 0.25) -> np.ndarray:
    """(num_classes+1) x (num_classes+1) matrix, rows = true class, columns =
    predicted class; index `num_classes` is "background" (no object /
    no detection). matrix[a][b] with a==b is a correct detection;
    matrix[bg][b] is a false positive of class b (no matching ground truth);
    matrix[a][bg] is a missed detection of class a; matrix[a][b] with a != b
    and both real classes is a genuine cross-class confusion -- a box the
    model localized correctly but classified wrong.

    Matching here is CLASS-AGNOSTIC greedy IoU: a prediction can match any
    ground-truth box regardless of class. This is deliberately different
    from evaluate_detections'/_micro_prf1's PER-CLASS matching above -- that
    per-class approach would score a D10 box predicted as D00 as a separate
    FN for D10 and a separate FP for D00, never revealing they were the same
    box. A confusion matrix's entire point is showing that connection, so it
    needs its own matching pass instead of reusing the per-class one.
    """
    bg = num_classes
    matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=int)

    for p, g in zip(preds, gts):
        keep = p["scores"] >= confidence_threshold
        pred_boxes, pred_labels = p["boxes"][keep], p["labels"][keep]
        pred_scores = p["scores"][keep]
        gt_boxes, gt_labels = g["boxes"], g["labels"]

        if len(gt_boxes) == 0:
            for lbl in pred_labels:
                matrix[bg, lbl] += 1
            continue
        if len(pred_boxes) == 0:
            for lbl in gt_labels:
                matrix[lbl, bg] += 1
            continue

        order = np.argsort(-pred_scores)
        pred_boxes, pred_labels = pred_boxes[order], pred_labels[order]
        ious = box_iou(pred_boxes, gt_boxes)
        claimed = np.zeros(len(gt_boxes), dtype=bool)

        for i, row in enumerate(ious):
            candidates = np.where((row >= iou_threshold) & (~claimed))[0]
            if len(candidates):
                best = candidates[np.argmax(row[candidates])]
                claimed[best] = True
                matrix[gt_labels[best], pred_labels[i]] += 1
            else:
                matrix[bg, pred_labels[i]] += 1

        for j in np.where(~claimed)[0]:
            matrix[gt_labels[j], bg] += 1

    return matrix


def evaluate_detections(preds, gts, num_classes: int = 4,
                        iou_start: float = 0.50, iou_end: float = 0.95,
                        iou_step: float = 0.05,
                        pr_iou_threshold: float = 0.50,
                        pr_confidence_threshold: float = 0.25) -> dict:
    """Compute the full metric suite required by the project spec section 5.3.

    Args:
        preds: list (per image) of dicts: boxes (N,4) xyxy abs, scores (N,), labels (N,) int
        gts:   list (per image) of dicts: boxes (M,4) xyxy abs, labels (M,) int
    Returns:
        dict with map50, map50_95, per_class_ap50, per_class_ap50_95,
        precision, recall, f1, and per-class ground-truth counts.
    """
    assert len(preds) == len(gts), "preds and gts must cover the same images, in the same order"

    # np.arange can drop the endpoint to floating point error; +half-step is safer.
    thresholds = np.arange(iou_start, iou_end + iou_step / 2, iou_step)

    # ap_table[class][threshold]
    ap_table = np.zeros((num_classes, len(thresholds)))
    gt_counts = np.zeros(num_classes, dtype=int)

    for class_id in range(num_classes):
        for t_idx, t in enumerate(thresholds):
            ap, num_gt = _ap_for_class(preds, gts, class_id, float(t))
            ap_table[class_id, t_idx] = ap
            gt_counts[class_id] = num_gt

    # Classes with zero ground truth are undefined, not zero -- averaging them in
    # would silently drag mAP down. Exclude them and report which were present.
    present = gt_counts > 0

    ap50_per_class = ap_table[:, 0]
    ap50_95_per_class = ap_table.mean(axis=1)

    map50 = float(ap50_per_class[present].mean()) if present.any() else 0.0
    map50_95 = float(ap50_95_per_class[present].mean()) if present.any() else 0.0

    prf1 = _micro_prf1(preds, gts, num_classes, pr_iou_threshold, pr_confidence_threshold)

    return {
        "map50": map50,
        "map50_95": map50_95,
        "per_class_ap50": {CLASS_NAMES[i]: float(ap50_per_class[i]) for i in range(num_classes)},
        "per_class_ap50_95": {CLASS_NAMES[i]: float(ap50_95_per_class[i]) for i in range(num_classes)},
        "gt_counts": {CLASS_NAMES[i]: int(gt_counts[i]) for i in range(num_classes)},
        "classes_present": [CLASS_NAMES[i] for i in range(num_classes) if present[i]],
        **prf1,
        "pr_iou_threshold": pr_iou_threshold,
        "pr_confidence_threshold": pr_confidence_threshold,
    }
