"""
Append-only experiment log (the project spec section 9: "No silent runs").

Every training or evaluation run adds one row to
experiments/results/experiment_log.csv, recording the config that produced the
numbers alongside the numbers themselves. The log is committed to git even
though model weights are not, so results stay traceable after weights are
deleted or regenerated.
"""

import csv
import hashlib
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = REPO_ROOT / "experiments" / "results" / "experiment_log.csv"

# Fixed column order. New columns must be appended at the end, never inserted
# in the middle, so previously written rows stay aligned with their header.
FIELDNAMES = [
    "timestamp",
    "run_id",
    "phase",
    "model",
    "config_hash",
    "dataset",
    "split",
    "seed",
    "epochs",
    "imgsz",
    "batch",
    "num_images",
    "map50",
    "map50_95",
    "ap_D00",
    "ap_D10",
    "ap_D20",
    "ap_D40",
    "precision",
    "recall",
    "f1",
    "latency_ms",
    "fps",
    # Parameter count is required for every run (the project spec section 5.3). Unlike
    # latency it is hardware-independent, so it stays comparable across machines.
    "param_count",
    "trainable_param_count",
    "train_runtime_sec",
    "eval_runtime_sec",
    "hardware",
    "weights_path",
    "notes",
]


def config_hash(config: dict) -> str:
    """Short stable hash of a config dict, so a row can be tied to exact settings.

    sort_keys=True makes the hash independent of dict ordering, so the same
    settings always produce the same hash across runs and machines.
    """
    blob = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def describe_hardware() -> str:
    """One-line hardware description, recorded so latency/FPS numbers are interpretable.

    Speed metrics are only comparable across rows measured on the same hardware
    (the project spec section 5.3 requires identical hardware for the model comparison).
    """
    parts = [platform.processor() or platform.machine()]
    try:
        import torch

        if torch.cuda.is_available():
            parts.append(f"CUDA:{torch.cuda.get_device_name(0)}")
        else:
            parts.append("CPU-only")
    except ImportError:
        parts.append("torch-not-installed")
    return " | ".join(p for p in parts if p)


def log_run(row: dict):
    """Append one run to the CSV, creating the file with a header if needed.

    Unknown keys are rejected rather than silently dropped, so a typo in a
    metric name fails loudly instead of quietly losing a result.
    """
    unknown = set(row) - set(FIELDNAMES)
    if unknown:
        raise ValueError(f"Unknown experiment_log columns: {sorted(unknown)}")

    row = dict(row)
    row.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))
    row.setdefault("hardware", describe_hardware())

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LOG_PATH.exists()

    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in FIELDNAMES})

    return LOG_PATH
