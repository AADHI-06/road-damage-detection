"""
Known-answer tests for src/eval/metrics.py.

A silently-wrong mAP implementation would invalidate every number in this
project, and unlike a training bug it produces plausible-looking output. These
cases are small enough to work out by hand, so the expected values below are
derived, not copied from the code's own output.

Run:
    python src/eval/test_metrics.py
"""

import numpy as np

from metrics import box_iou, evaluate_detections


def approx(a, b, tol=1e-6):
    return abs(a - b) < tol


def test_box_iou():
    identical = box_iou(np.array([[0, 0, 10, 10.0]]), np.array([[0, 0, 10, 10.0]]))
    assert approx(identical[0, 0], 1.0), identical

    # Overlap corner-to-corner: intersection 5x5=25, union 100+100-25=175.
    partial = box_iou(np.array([[0, 0, 10, 10.0]]), np.array([[5, 5, 15, 15.0]]))
    assert approx(partial[0, 0], 25 / 175), partial

    disjoint = box_iou(np.array([[0, 0, 10, 10.0]]), np.array([[50, 50, 60, 60.0]]))
    assert approx(disjoint[0, 0], 0.0), disjoint

    empty = box_iou(np.zeros((0, 4)), np.array([[0, 0, 1, 1.0]]))
    assert empty.shape == (0, 1), empty.shape
    print("  box_iou: OK")


def test_perfect_predictions():
    """Predictions identical to ground truth => AP and F1 are exactly 1.0."""
    gt_boxes = np.array([[0, 0, 10, 10.0], [20, 20, 30, 30.0]])
    gts = [{"boxes": gt_boxes, "labels": np.array([0, 0])}]
    preds = [{"boxes": gt_boxes, "scores": np.array([0.9, 0.9]), "labels": np.array([0, 0])}]

    r = evaluate_detections(preds, gts, num_classes=4)
    assert approx(r["map50"], 1.0), r["map50"]
    assert approx(r["f1"], 1.0), r["f1"]
    # Only class D00 has ground truth; the other three must be excluded, not counted as 0.
    assert r["classes_present"] == ["D00"], r["classes_present"]
    print("  perfect predictions: OK")


def test_no_predictions():
    gts = [{"boxes": np.array([[0, 0, 10, 10.0]]), "labels": np.array([0])}]
    preds = [{"boxes": np.zeros((0, 4)), "scores": np.zeros(0), "labels": np.zeros(0, dtype=int)}]

    r = evaluate_detections(preds, gts, num_classes=4)
    assert approx(r["map50"], 0.0), r["map50"]
    assert approx(r["recall"], 0.0), r["recall"]
    assert r["fn"] == 1, r["fn"]
    print("  no predictions: OK")


def test_hand_computed_ap():
    """Two GT, three predictions (TP, FP, TP) -> AP = 5/6 by hand.

    PR curve walking predictions in descending confidence:
      after p1 (TP): P=1/1=1.00, R=1/2=0.5
      after p2 (FP): P=1/2=0.50, R=1/2=0.5
      after p3 (TP): P=2/3=0.67, R=2/2=1.0
    After making precision monotonically non-increasing from the right,
    the curve is 1.0 up to recall 0.5 and 2/3 from 0.5 to 1.0:
      AP = 0.5*1.0 + 0.5*(2/3) = 5/6
    """
    gts = [{
        "boxes": np.array([[0, 0, 10, 10.0], [100, 100, 110, 110.0]]),
        "labels": np.array([0, 0]),
    }]
    preds = [{
        "boxes": np.array([
            [0, 0, 10, 10.0],        # exact match for GT1  -> TP
            [500, 500, 510, 510.0],  # matches nothing      -> FP
            [100, 100, 110, 110.0],  # exact match for GT2  -> TP
        ]),
        "scores": np.array([0.9, 0.8, 0.7]),
        "labels": np.array([0, 0, 0]),
    }]

    r = evaluate_detections(preds, gts, num_classes=4)
    assert approx(r["per_class_ap50"]["D00"], 5 / 6), r["per_class_ap50"]

    # P/R/F1 at conf 0.25: all three predictions survive -> TP=2, FP=1, FN=0.
    assert r["tp"] == 2 and r["fp"] == 1 and r["fn"] == 0, (r["tp"], r["fp"], r["fn"])
    assert approx(r["precision"], 2 / 3), r["precision"]
    assert approx(r["recall"], 1.0), r["recall"]
    assert approx(r["f1"], 0.8), r["f1"]
    print("  hand-computed AP = 5/6: OK")


def test_duplicate_detection_is_false_positive():
    """Two predictions on one GT: the higher-scoring one is the TP, the other an FP."""
    gts = [{"boxes": np.array([[0, 0, 10, 10.0]]), "labels": np.array([0])}]
    preds = [{
        "boxes": np.array([[0, 0, 10, 10.0], [0, 0, 10, 10.0]]),
        "scores": np.array([0.9, 0.8]),
        "labels": np.array([0, 0]),
    }]

    r = evaluate_detections(preds, gts, num_classes=4)
    assert r["tp"] == 1 and r["fp"] == 1, (r["tp"], r["fp"])
    # Recall is perfect (the object was found) but precision is halved.
    assert approx(r["recall"], 1.0), r["recall"]
    assert approx(r["precision"], 0.5), r["precision"]
    print("  duplicate detection counted as FP: OK")


def test_class_confusion_is_not_credited():
    """A perfectly-placed box with the WRONG class label is an FP, not a TP."""
    gts = [{"boxes": np.array([[0, 0, 10, 10.0]]), "labels": np.array([0])}]
    preds = [{
        "boxes": np.array([[0, 0, 10, 10.0]]),
        "scores": np.array([0.9]),
        "labels": np.array([3]),  # D40 predicted where D00 is annotated
    }]

    r = evaluate_detections(preds, gts, num_classes=4)
    assert r["tp"] == 0 and r["fp"] == 1 and r["fn"] == 1, (r["tp"], r["fp"], r["fn"])
    assert approx(r["map50"], 0.0), r["map50"]
    print("  wrong-class prediction counted as FP: OK")


def test_iou_threshold_sensitivity():
    """A loose box scores at IoU 0.5 but not at IoU 0.95 -> map50 > map50_95."""
    gts = [{"boxes": np.array([[0, 0, 100, 100.0]]), "labels": np.array([0])}]
    preds = [{
        "boxes": np.array([[10, 10, 100, 100.0]]),  # IoU = 8100/10000 = 0.81
        "scores": np.array([0.9]),
        "labels": np.array([0]),
    }]

    r = evaluate_detections(preds, gts, num_classes=4)
    assert approx(r["map50"], 1.0), r["map50"]
    # Thresholds 0.50..0.80 match (7 of 10), 0.85..0.95 do not.
    assert approx(r["map50_95"], 0.7), r["map50_95"]
    print("  IoU threshold sensitivity: OK")


if __name__ == "__main__":
    print("Running metrics known-answer tests...")
    test_box_iou()
    test_perfect_predictions()
    test_no_predictions()
    test_hand_computed_ap()
    test_duplicate_detection_is_false_positive()
    test_class_confusion_is_not_credited()
    test_iou_threshold_sensitivity()
    print("\nAll metric tests passed.")
