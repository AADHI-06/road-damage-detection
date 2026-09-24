#!/usr/bin/env bash
# Phase 3a: train both plain baselines on SOURCE and evaluate in-domain.
# No attention module -- that is Phase 3b (see the project spec section 3).
#
# Set DEVICE=0 (or another CUDA index) when a GPU is available; the default
# "cpu" is only practical for the smoke test below.
set -e
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cpu}"

# CHUNK_EPOCHS limits how many epochs each invocation runs before stopping cleanly.
# Leave unset to train straight through. Both scripts resume automatically, so
# rerunning this script continues from the last completed epoch.
YOLO_CHUNK="${YOLO_CHUNK:-}"
RCNN_CHUNK="${RCNN_CHUNK:-}"

python src/models/train_yolo.py \
    --data config/dataset_source.yaml \
    --name yolov8n_source \
    --device "$DEVICE" \
    ${YOLO_CHUNK:+--chunk-epochs "$YOLO_CHUNK"}

python src/models/train_faster_rcnn.py \
    --data-root data/processed/source \
    --name faster_rcnn_source \
    --device "$DEVICE" \
    ${RCNN_CHUNK:+--chunk-epochs "$RCNN_CHUNK"}
