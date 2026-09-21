#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

python src/data/verify_dataset.py
python src/data/dataset_stats.py
