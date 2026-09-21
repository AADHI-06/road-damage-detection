#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

python src/data/build_splits.py
python src/data/sanity_check_labels.py
