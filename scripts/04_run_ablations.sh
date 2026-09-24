#!/usr/bin/env bash
# Phase 5: hyperparameter ablations (the project spec section 5.4).
# Backbone size, augmentation on/off, input resolution -- YOLOv8-only.
# Requires Phase 3a's yolov8n_source to already be trained (reused as the
# baseline reference point for every axis).
set -e
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cpu}"
CHUNK_EPOCHS="${CHUNK_EPOCHS:-}"

python src/ablation/run_ablation.py \
    --device "$DEVICE" \
    ${CHUNK_EPOCHS:+--chunk-epochs "$CHUNK_EPOCHS"}
