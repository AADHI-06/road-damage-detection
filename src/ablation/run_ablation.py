"""
Phase 5: hyperparameter ablations (the project spec section 5.4).

Three ablation axes, YOLOv8-only. Backbone size and coordinate attention are
architectural properties of YOLOv8 specifically; Faster R-CNN reproduces
DA-RDD's fixed detector backbone (the project spec section 3) and is not itself an
ablation subject.

  1. Backbone size:       YOLOv8n (Phase 3a baseline) vs YOLOv8s vs YOLOv8m
  2. Data augmentation:   on (Phase 3a baseline) vs off
  3. Input resolution:    640 (Phase 3a baseline) vs 416 vs 960
  4. Coordinate attention: with vs without -- PHASE 3b ONLY, not run by this
     script. It needs an actual trained 3b checkpoint (train_yolo.py
     --attention) to exist first; compare its logged row against the 3a
     baseline row by hand once both exist, the same way this script compares
     each of ITS variants against the reused 3a baseline row.

the project spec section 5.4: "Each ablation changes ONE variable at a time. Keep
everything else fixed." Concretely: every variant below is produced by
train_yolo.py using config/hyperparams.yaml's Phase 3a settings UNCHANGED
except the single studied flag, and the Phase 3a baseline run itself
(yolov8n_source, already trained) is REUSED as the reference point for all
three axes -- it is never retrained here.

COMPUTE NOTE: matching the 3a baseline's 100-epoch schedule exactly (the
faithful choice) costs roughly 2h/variant on a T4 -- 5 new variants is ~10
GPU-hours. --epochs shortens this deliberately; doing so is a real,
reportable deviation from the 3a schedule (not a free lunch), so it prints a
warning that must be echoed in the report if used.

This script ORCHESTRATES; it does not reimplement training. Each variant is a
subprocess call to src/models/train_yolo.py, so ablations reuse the exact
same chunking/resume/eval/logging machinery Phase 3a and 3b already use --
rerunning this script resumes whatever variant didn't finish, the same way
rerunning a train_yolo.py chunk command does.

Usage:
    python src/ablation/run_ablation.py --dry-run              # print the plan, run nothing
    python src/ablation/run_ablation.py                        # all axes, full 100-epoch budget
    python src/ablation/run_ablation.py --axes backbone         # one axis only
    python src/ablation/run_ablation.py --epochs 30             # shortened budget (see note above)
    python src/ablation/run_ablation.py --chunk-epochs 20 --device 0   # Colab, chunked
    python src/ablation/run_ablation.py --smoke                 # tiny subset, pipeline check

Writes:
    experiments/results/ablation_results.csv   -- one row per variant, plus
                                                    the reused 3a baseline row
    (appends to) experiments/results/experiment_log.csv, via train_yolo.py
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_YOLO = REPO_ROOT / "src" / "models" / "train_yolo.py"
RESULTS_DIR = REPO_ROOT / "experiments" / "results"
LOG_PATH = RESULTS_DIR / "experiment_log.csv"

BASELINE_RUN_ID = "yolov8n_source"  # Phase 3a, already trained -- reused, not retrained

# Each variant: {axis, suffix, flags (extra CLI args to train_yolo.py), describe (for the printed plan)}
VARIANTS = [
    {"axis": "backbone", "suffix": "backbone-s", "flags": ["--base-model", "yolov8s.pt"],
     "describe": "YOLOv8s (vs. 3a's YOLOv8n)"},
    {"axis": "backbone", "suffix": "backbone-m", "flags": ["--base-model", "yolov8m.pt"],
     "describe": "YOLOv8m (vs. 3a's YOLOv8n)"},
    {"axis": "augment", "suffix": "augment-off", "flags": ["--no-augment"],
     "describe": "augmentation OFF (vs. 3a's ON)"},
    {"axis": "resolution", "suffix": "resolution-416", "flags": ["--imgsz", "416"],
     "describe": "imgsz=416 (vs. 3a's 640)"},
    {"axis": "resolution", "suffix": "resolution-960", "flags": ["--imgsz", "960"],
     "describe": "imgsz=960 (vs. 3a's 640)"},
]

RESULT_FIELDNAMES = [
    "axis", "variant", "run_id", "model", "imgsz", "augment", "epochs",
    "map50", "map50_95", "f1", "param_count",
    "map50_delta_vs_baseline", "map50_pct_change_vs_baseline",
]


def read_experiment_log() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    with open(LOG_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def latest_row_for(rows: list[dict], run_id: str) -> dict | None:
    """Most recent experiment_log.csv row for a run_id (a run can have been
    evaluated more than once, e.g. after a resumed final chunk -- the latest
    row is the current, trustworthy one)."""
    matches = [r for r in rows if r["run_id"] == run_id]
    if not matches:
        return None
    return max(matches, key=lambda r: r["timestamp"])


def build_command(variant: dict, args) -> list[str]:
    run_id = f"yolov8n_ablation_{variant['suffix']}"
    cmd = [
        sys.executable, str(TRAIN_YOLO),
        "--data", str(REPO_ROOT / "config" /
                     ("dataset_source_subset.yaml" if args.smoke else "dataset_source.yaml")),
        "--name", run_id,
        "--phase-label", "5",
        "--device", args.device,
    ]
    if args.epochs is not None:
        cmd += ["--epochs", str(args.epochs)]
    if args.chunk_epochs is not None:
        cmd += ["--chunk-epochs", str(args.chunk_epochs)]
    if args.smoke:
        cmd += ["--smoke"]
    cmd += variant["flags"]
    return cmd, run_id


def pct_change(new: float, base: float):
    if base == 0:
        return ""
    return round(100.0 * (new - base) / base, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axes", nargs="+", choices=["backbone", "augment", "resolution"],
                    default=["backbone", "augment", "resolution"])
    ap.add_argument("--epochs", type=int, default=None,
                    help="TOTAL epochs per variant. Default: hyperparams.yaml's 100 (matches "
                         "the 3a baseline schedule exactly). Lowering this is a real, "
                         "reportable deviation -- see the module docstring's compute note.")
    ap.add_argument("--chunk-epochs", type=int, default=None,
                    help="train at most this many epochs per invocation, per variant, then stop "
                         "cleanly -- passed straight through to train_yolo.py. Rerun this script "
                         "to resume whichever variants aren't finished yet.")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny-subset pipeline check (source_subset, --smoke on train_yolo.py). "
                         "Not reportable; use before committing to the real ablation grid.")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    args = ap.parse_args()

    if args.epochs is not None and args.epochs < 100 and not args.smoke:
        print(f"WARNING: --epochs {args.epochs} is fewer than the Phase 3a baseline's 100. "
              f"This is a real deviation from 'identical hyperparameters' -- state it explicitly "
              f"in the report's ablation section, don't present these as directly comparable to "
              f"a full 100-epoch run.\n")

    plan = [v for v in VARIANTS if v["axis"] in args.axes]

    print("Ablation plan:")
    for v in plan:
        run_id = f"yolov8n_ablation_{v['suffix']}"
        print(f"  [{v['axis']:10}] {run_id:32} {v['describe']}")
    print(f"  baseline (reused, not retrained): {BASELINE_RUN_ID}\n")

    if args.dry_run:
        for v in plan:
            cmd, _ = build_command(v, args)
            print("  $ " + " ".join(cmd))
        return

    for v in plan:
        cmd, run_id = build_command(v, args)
        print(f"\n{'=' * 70}\nRunning: {run_id}  ({v['describe']})\n{'=' * 70}")
        print("  $ " + " ".join(cmd))
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\nFAILED: {run_id} exited with code {result.returncode}. "
                  f"Fix and rerun this script -- it will resume from wherever "
                  f"training left off (same as any other train_yolo.py chunk).")
            sys.exit(result.returncode)

    if args.smoke:
        print("\nSMOKE TEST run -- pipeline validated, no ablation_results.csv written. "
              "Rerun without --smoke for the real grid.")
        return

    # Build the comparison table from experiment_log.csv, not from anything
    # held in memory -- this makes rerunning just the summary step (without
    # retraining) trivial, and matches how cross_country_eval.py works.
    rows = read_experiment_log()
    baseline = latest_row_for(rows, BASELINE_RUN_ID)
    if baseline is None:
        print(f"\nWARNING: no experiment_log.csv row found for baseline run "
              f"'{BASELINE_RUN_ID}'. Train Phase 3a first -- writing ablation "
              f"variant rows without a baseline to compare against.")

    out_rows = []
    if baseline is not None:
        out_rows.append({
            "axis": "baseline", "variant": "yolov8n / augment-on / imgsz-640",
            "run_id": BASELINE_RUN_ID, "model": baseline["model"],
            "imgsz": baseline["imgsz"], "augment": "on", "epochs": baseline["epochs"],
            "map50": baseline["map50"], "map50_95": baseline["map50_95"],
            "f1": baseline["f1"], "param_count": baseline["param_count"],
            "map50_delta_vs_baseline": 0.0, "map50_pct_change_vs_baseline": 0.0,
        })

    for v in plan:
        run_id = f"yolov8n_ablation_{v['suffix']}"
        row = latest_row_for(rows, run_id)
        if row is None:
            print(f"  (no experiment_log.csv row for {run_id} yet -- skipping in summary)")
            continue
        map50 = float(row["map50"])
        out_row = {
            "axis": v["axis"], "variant": v["describe"], "run_id": run_id,
            "model": row["model"], "imgsz": row["imgsz"],
            "augment": "off" if v["axis"] == "augment" else "on",
            "epochs": row["epochs"], "map50": row["map50"], "map50_95": row["map50_95"],
            "f1": row["f1"], "param_count": row["param_count"],
        }
        if baseline is not None:
            base_map50 = float(baseline["map50"])
            out_row["map50_delta_vs_baseline"] = round(map50 - base_map50, 4)
            out_row["map50_pct_change_vs_baseline"] = pct_change(map50, base_map50)
        else:
            out_row["map50_delta_vs_baseline"] = ""
            out_row["map50_pct_change_vs_baseline"] = ""
        out_rows.append(out_row)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "ablation_results.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\nWrote {out_path} ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
