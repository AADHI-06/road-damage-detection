"""
Phase 4 -- THE core experiment (CLAUDE.md section 5.2, section 7 Phase 4).

Loads each trained Phase 3a baseline UNMODIFIED and evaluates it on:
  - SOURCE test        (in-domain reference point, India + Japan)
  - each TARGET country, separately, zero-shot (never seen during training):
    Czech, Norway, United_States, China_MotorBike

Produces the degradation table -- model x country x metric -- and a handful of
qualitative failure images per model/country. Results are reported PER COUNTRY
and never averaged away (section 5.2: "the variation between countries is
itself a result").

DELIBERATE DEVIATIONS FROM THE FULL EVALUATION, all compute-driven and time-
boxed, and ALL MUST BE STATED wherever these numbers are used -- none of them
are permanent; DATASET_SAMPLE_SIZES is a one-line change to revert:

  - China_Drone is EXCLUDED from this script's target list. YOLOv8 already has
    a complete China_Drone result logged (run_id yolo_source_eval_on_china_drone,
    2,401 images, mAP@0.5=0.2107) from before this exclusion -- that row is
    NOT deleted, but no model evaluated here from now on will have a matching
    China_Drone number. Any model x country table built from this script's
    output is therefore asymmetric (5 target countries for YOLOv8, 4 for
    whatever runs after this change) unless that's corrected by hand.

  - Norway, US, and China_MotorBike are each evaluated on a SEEDED RANDOM
    SUBSAMPLE of only 1,500 images (DATASET_SAMPLE_SIZES), not their full sets
    (8,161 / 4,805 / 1,977 respectively) -- a deliberate, time-boxed reduction
    to get a first Faster R-CNN cross-country reading quickly under a tight
    deadline. THIS IS PRELIMINARY. 1,500 images is still small enough that
    mAP on it carries real sampling noise (worse for China_MotorBike, whose
    full set is only 1,977 -- this subsample is most of it; better for Norway,
    where 1,500 of 8,161 is a much thinner slice) -- treat these numbers as
    directional, not final, and re-run at full scale (delete the three entries
    from DATASET_SAMPLE_SIZES, or override per-run) before they go in the report.
    The subsamples are reproducible (fixed seed) but are NOT the same images
    YOLOv8's existing full-set results were computed on, so a direct
    model x model comparison on these three countries mixes a full-set number
    (YOLOv8) with a 250-image subsampled one (Faster R-CNN) unless YOLOv8 is
    also re-run on the identical subsample. See get_predictions_and_gt's
    `sample_size` parameter in evaluate.py for how the subsample is drawn.

Three comparisons this table supports (section 5.2):
  1. Faster R-CNN vs. YOLOv8 -> paradigm trade-off
  2. in-domain vs. zero-shot, per model -> the generalization gap
  3. which architecture degrades LEAST under domain shift -> the headline finding

Usage:
    python src/eval/cross_country_eval.py                      # both models, all countries
    python src/eval/cross_country_eval.py --models yolo         # one model only
    python src/eval/cross_country_eval.py --datasets czech norway
    python src/eval/cross_country_eval.py --skip-failure-examples   # metrics only, faster
    python src/eval/cross_country_eval.py --smoke               # tiny subset, pipeline check

Writes:
    experiments/results/cross_country_results.csv
    experiments/results/figures/failure_examples/<model>/<country>/*.jpg
    (appends to) experiments/results/experiment_log.csv
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "eval"))
sys.path.insert(0, str(REPO_ROOT / "src" / "models"))
sys.path.insert(0, str(REPO_ROOT / "src" / "utils"))

from evaluate import (  # noqa: E402
    get_predictions_and_gt, assert_has_ground_truth, load_hyperparams, labels_dir_for,
)
from metrics import evaluate_detections, CLASS_NAMES  # noqa: E402
from experiment_logger import log_run, config_hash  # noqa: E402
from visualize import select_failure_examples, draw_gt_and_predictions  # noqa: E402
from benchmark_speed import load_model, count_parameters  # noqa: E402

RUNS_DIR = REPO_ROOT / "experiments" / "runs"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "experiments" / "results"
FAILURE_DIR = RESULTS_DIR / "figures" / "failure_examples"
LOG_PATH = RESULTS_DIR / "experiment_log.csv"

# The Phase 3a baselines. Weight paths match scripts/02_train_baselines.sh's
# --name arguments (and the yolo26 handoff merge); if a run was renamed, update
# here (or pass --weights-override).
#
# "family" is what drives every downstream dispatch (which loader, which imgsz,
# which predict path) -- evaluate.py / benchmark_speed.py branch on the string
# "yolo" vs "faster_rcnn", and yolo26 rides the identical ultralytics code path
# as yolov8n. The dict KEY ("yolo" / "yolo26" / "faster_rcnn") stays the model's
# own identity, used for run_id, --models selection, and logging.
#
# "train_config" names the hyperparams.yaml slice (plus any per-model overrides)
# whose hash is recorded on every Phase 4 row for this model -- it must match the
# config the Phase 3a checkpoint was actually trained with, so a row is always
# traceable back to its weights. yolo26 trained with model=yolo26n.pt, batch=8
# (see experiment_log.csv row yolo26n_source, config_hash 8d3939078f40).
MODELS = {
    "yolo": {
        "weights": RUNS_DIR / "yolov8n_source" / "weights" / "best.pt",
        "model_name": "yolov8n",
        "family": "yolo",
        "train_config_overrides": {},
    },
    "yolo26": {
        "weights": RUNS_DIR / "yolo26n_source" / "weights" / "best.pt",
        "model_name": "yolo26n",
        "family": "yolo",
        "train_config_overrides": {"model": "yolo26n.pt", "batch": 8},
    },
    "faster_rcnn": {
        "weights": RUNS_DIR / "faster_rcnn_source" / "best.pt",
        "model_name": "faster_rcnn_resnet50_fpn",
        "family": "faster_rcnn",
        "train_config_overrides": {},
    },
}

# In-domain reference + every zero-shot target. "kind" distinguishes them in
# the output table; SOURCE is what every TARGET number is measured against.
DATASETS = {
    "source": {
        "images": PROCESSED_DIR / "source" / "images" / "test",
        "labels": PROCESSED_DIR / "source" / "labels" / "test",
        "kind": "in-domain",
    },
    "czech": {
        "images": PROCESSED_DIR / "target" / "czech" / "images",
        "labels": PROCESSED_DIR / "target" / "czech" / "labels",
        "kind": "zero-shot",
    },
    "norway": {
        "images": PROCESSED_DIR / "target" / "norway" / "images",
        "labels": PROCESSED_DIR / "target" / "norway" / "labels",
        "kind": "zero-shot",
    },
    "us": {
        "images": PROCESSED_DIR / "target" / "us" / "images",
        "labels": PROCESSED_DIR / "target" / "us" / "labels",
        "kind": "zero-shot",
    },
    "china_motorbike": {
        "images": PROCESSED_DIR / "target" / "china_motorbike" / "images",
        "labels": PROCESSED_DIR / "target" / "china_motorbike" / "labels",
        "kind": "zero-shot",
    },
}

# Per-dataset random-subsample cap (see module docstring).
# *** PRELIMINARY / TIME-BOXED -- capped at 1,500 for a fast first Faster R-CNN
# reading under a tight deadline. Revisit before the report: delete entries here
# (or raise the numbers) and re-run for the real, full-scale result. *** SEED
# matches the project's global reproducibility seed (config/hyperparams.yaml).
#
# APPLIES TO FASTER R-CNN ONLY (see SUBSAMPLE_FAMILIES). The YOLO-family models
# (yolov8n, yolo26n) are ~15x lighter and always run the FULL target sets on
# CPU in well under the Faster R-CNN budget, so subsampling them would only
# throw away precision AND make yolo-vs-yolo comparisons inconsistent (yolov8n's
# existing Phase 4 rows are all full-set).
DATASET_SAMPLE_SIZES = {"norway": 1500, "us": 1500, "china_motorbike": 1500}
SAMPLE_SEED = 42
SUBSAMPLE_FAMILIES = {"faster_rcnn"}

RESULT_FIELDNAMES = [
    "model", "dataset", "kind", "num_images", "num_gt_boxes",
    "map50", "map50_95",
    "ap_D00", "ap_D10", "ap_D20", "ap_D40",
    "precision", "recall", "f1",
    "map50_delta_vs_indomain", "map50_pct_change_vs_indomain",
    "f1_delta_vs_indomain", "f1_pct_change_vs_indomain",
    "eval_runtime_sec",
]


def evaluate_one(family: str, weights: Path, images_dir: Path, labels_dir: Path,
                 hp: dict, limit: int | None = None, sample_size: int | None = None,
                 device: str = "cpu") -> dict:
    """Run inference + score one (model, dataset) pair. Returns metrics dict
    plus the raw per-image preds/gts/paths (for failure-example selection).

    `limit` is passed through to get_predictions_and_gt so it slices BEFORE
    inference runs (a --smoke run must actually be fast, not just report a
    small number after doing the full-dataset work anyway).

    `sample_size` is the Norway-style seeded random subsample (see module
    docstring) -- distinct from `limit`: `limit` is a deterministic "first N"
    used only for --smoke pipeline checks, `sample_size` is a real (if
    reduced) reportable evaluation and must be randomized to stay
    representative.

    `device` is forwarded all the way to the actual inference calls --
    without it, --device cuda had no effect on the bulk evaluation here (see
    evaluate.py's get_predictions_and_gt docstring for the bug this fixed).
    """
    image_paths, preds, gts = get_predictions_and_gt(
        family, weights, images_dir, labels_dir, hp, limit=limit,
        sample_size=sample_size, sample_seed=SAMPLE_SEED, device=device)

    num_gt = assert_has_ground_truth(gts, labels_dir, len(image_paths))

    ev = hp["eval"]
    metrics = evaluate_detections(
        preds, gts,
        num_classes=len(CLASS_NAMES),
        iou_start=ev["iou_thresholds_start"],
        iou_end=ev["iou_thresholds_end"],
        iou_step=ev["iou_thresholds_step"],
        pr_iou_threshold=ev["pr_iou_threshold"],
        pr_confidence_threshold=ev["pr_confidence_threshold"],
    )
    metrics["num_images"] = len(image_paths)
    metrics["num_gt_boxes"] = num_gt
    return metrics, image_paths, preds, gts


def pct_change(new: float, base: float) -> float | str:
    """(new - base) / base, as a percentage. '' (not 0 or NaN) when the
    in-domain baseline itself is 0 -- a percentage change from zero is
    undefined, and printing 0.0 there would misleadingly read as 'no change'.
    """
    if base == 0:
        return ""
    return round(100.0 * (new - base) / base, 2)


def read_experiment_log() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    with open(LOG_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_logged_run(rows: list[dict], run_id: str, phase: str, expected_hash: str) -> dict | None:
    """Most recent experiment_log.csv row for this exact (run_id, phase,
    config_hash), or None. The hash check matters: a stale row from a
    DIFFERENT hyperparams.yaml (e.g. before an edit) must not be silently
    reused as if it were current."""
    matches = [r for r in rows if r["run_id"] == run_id and r["phase"] == phase
              and r["config_hash"] == expected_hash]
    if not matches:
        return None
    return max(matches, key=lambda r: r["timestamp"])


def row_from_log_entry(entry: dict, kind: str) -> dict:
    """Reconstruct a cross_country_results.csv row from an experiment_log.csv
    row, for a (model, dataset) pair that's being skipped because it already
    ran successfully in a previous (crashed or completed) invocation."""
    return {
        "model": entry["model"], "dataset": entry["dataset"], "kind": kind,
        "num_images": entry["num_images"], "num_gt_boxes": "",  # not stored in experiment_log
        "map50": entry["map50"], "map50_95": entry["map50_95"],
        "ap_D00": entry["ap_D00"], "ap_D10": entry["ap_D10"],
        "ap_D20": entry["ap_D20"], "ap_D40": entry["ap_D40"],
        "precision": entry["precision"], "recall": entry["recall"], "f1": entry["f1"],
        "eval_runtime_sec": entry["eval_runtime_sec"],
    }


def write_results_csv(rows: list[dict], out_path: Path):
    """Write the given rows to the CSV, overwriting whatever was there.
    Callers must pass the FULL row set to keep, not just this invocation's
    -- see rebuild_full_results_csv, which is what main() actually uses.
    """
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def rebuild_full_results_csv(phase: str, out_path: Path):
    """Rebuild cross_country_results.csv from ALL matching rows currently in
    experiment_log.csv -- not just the ones THIS invocation evaluated.

    experiment_log.csv is append-only and authoritative; this CSV is only a
    materialized, human-readable view of it (same pattern run_ablation.py and
    generate_report_tables.py already use). Without rebuilding from the full
    log every time, running `--models faster_rcnn` after an earlier
    `--models yolo` run would overwrite the file with faster_rcnn-only rows
    and silently discard yolo's -- exactly what happened before this fix
    (yolo had 6 rows logged; a later faster_rcnn-only invocation wrote a
    2-row file, and yolo's rows were only ever recoverable from
    experiment_log.csv, not from this CSV).

    Called after every dataset (not just at the end) so a crash loses at
    most one dataset's results, matching the crash-recovery design already
    in place for individual (model, dataset) resume-skip.
    """
    log_rows = read_experiment_log()
    matches = [r for r in log_rows if r["phase"] == phase
              and "_source_eval_on_" in r["run_id"]]

    latest = {}
    for r in matches:
        key = (r["model"], r["dataset"])
        ts = r.get("timestamp", "")
        if key not in latest or ts >= latest[key][0]:
            latest[key] = (ts, r)

    by_model: dict[str, dict[str, dict]] = {}
    for (model, dataset), (_, r) in latest.items():
        by_model.setdefault(model, {})[dataset] = r

    out_rows = []
    for model, by_dataset in by_model.items():
        indomain = by_dataset.get("source")
        indomain_map50 = float(indomain["map50"]) if indomain else None
        indomain_f1 = float(indomain["f1"]) if indomain else None

        for dataset, r in by_dataset.items():
            row = row_from_log_entry(r, "in-domain" if dataset == "source" else "zero-shot")
            row["eval_runtime_sec"] = r.get("eval_runtime_sec", "")
            if indomain_map50 is not None:
                map50, f1 = float(r["map50"]), float(r["f1"])
                row["map50_delta_vs_indomain"] = round(map50 - indomain_map50, 4)
                row["map50_pct_change_vs_indomain"] = pct_change(map50, indomain_map50)
                row["f1_delta_vs_indomain"] = round(f1 - indomain_f1, 4)
                row["f1_pct_change_vs_indomain"] = pct_change(f1, indomain_f1)
            else:
                for k in ("map50_delta_vs_indomain", "map50_pct_change_vs_indomain",
                         "f1_delta_vs_indomain", "f1_pct_change_vs_indomain"):
                    row[k] = ""
            out_rows.append(row)

    # Stable, readable ordering: grouped by model, source first within each.
    out_rows.sort(key=lambda r: (r["model"], r["dataset"] != "source", r["dataset"]))
    write_results_csv(out_rows, out_path)
    return out_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    ap.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    ap.add_argument("--skip-failure-examples", action="store_true")
    ap.add_argument("--failure-examples-per-country", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--smoke", action="store_true",
                    help="evaluate only the first 30 images of each dataset -- "
                         "pipeline check, NOT reportable results")
    ap.add_argument("--force", action="store_true",
                    help="re-evaluate every (model, dataset) pair even if a matching "
                         "experiment_log.csv row already exists. Default is to skip and "
                         "reuse it -- this is what makes rerunning this script after a "
                         "crash a genuine resume instead of starting over.")
    args = ap.parse_args()

    hp = load_hyperparams()
    limit = 30 if args.smoke else None
    phase = "4-smoke" if args.smoke else "4"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / ("cross_country_results_smoke.csv" if args.smoke
                              else "cross_country_results.csv")

    for model_kind in args.models:
        cfg = MODELS[model_kind]
        family = cfg["family"]
        weights = cfg["weights"]
        if not weights.exists():
            print(f"SKIP {model_kind}: weights not found at {weights} (train Phase 3a first)")
            continue

        # Everything below dispatches on `family` (which ultralytics/torchvision
        # code path), never on `model_kind` -- yolo26 shares yolov8n's path.
        base_cfg = hp["yolo"] if family == "yolo" else hp["faster_rcnn"]
        imgsz = base_cfg["imgsz"]

        # Parameter count once per model (architecture-only, cheap, doesn't
        # need re-measuring per country).
        _, module = load_model(family, weights, imgsz, args.device)
        params = count_parameters(module)

        # Hash the config this model's Phase 3a checkpoint was trained with
        # (base slice + per-model overrides), so every Phase 4 row traces back
        # to its exact weights. See MODELS["...train_config_overrides"].
        train_cfg = dict(base_cfg)
        train_cfg.update(cfg["train_config_overrides"])
        run_config_hash = config_hash(train_cfg)
        log_rows = read_experiment_log() if not args.force else []

        for dataset_key in args.datasets:
            d = DATASETS[dataset_key]
            run_id = f"{model_kind}_source_eval_on_{dataset_key}"
            print(f"\n=== {model_kind} on {dataset_key} ({d['kind']}) ===")

            existing = None if args.force else find_logged_run(log_rows, run_id, phase, run_config_hash)
            if existing is not None:
                print(f"  already logged (run_id={run_id}, matching config_hash) -- "
                     f"skipping re-evaluation. Use --force to redo it.")
                print(f"  mAP@0.5={existing['map50']}, F1={existing['f1']} (from experiment_log.csv)")
                # Rebuild from the full log even on a skip: keeps the CSV
                # complete (all models/datasets logged so far) regardless of
                # which --models/--datasets subset THIS invocation touches.
                rebuild_full_results_csv(phase, out_path)
                continue

            # Subsampling is a Faster R-CNN time-box only; YOLO-family models
            # always run the full set (see DATASET_SAMPLE_SIZES comment).
            sample_size = (DATASET_SAMPLE_SIZES.get(dataset_key)
                           if family in SUBSAMPLE_FAMILIES else None)
            t0 = time.perf_counter()
            metrics, image_paths, preds, gts = evaluate_one(
                family, weights, d["images"], d["labels"], hp, limit=limit,
                sample_size=sample_size, device=args.device)
            runtime = time.perf_counter() - t0
            if sample_size is not None:
                print(f"  (seeded random subsample: {sample_size} of the full set, seed={SAMPLE_SEED})")
            print(f"  {metrics['num_images']} images, {metrics['num_gt_boxes']} GT boxes, "
                  f"mAP@0.5={metrics['map50']:.4f}, F1={metrics['f1']:.4f}  ({runtime:.1f}s)")

            # One row per (model, dataset) in the master experiment log too --
            # CLAUDE.md section 9: "No silent runs." Also what makes the
            # skip-if-already-logged resume above possible.
            log_run({
                "run_id": run_id,
                "phase": phase,
                "model": cfg["model_name"],
                "config_hash": run_config_hash,
                "dataset": dataset_key,
                "split": "test" if dataset_key == "source" else "all",
                "seed": hp["seed"],
                "num_images": metrics["num_images"],
                "map50": round(metrics["map50"], 4),
                "map50_95": round(metrics["map50_95"], 4),
                "ap_D00": round(metrics["per_class_ap50"]["D00"], 4),
                "ap_D10": round(metrics["per_class_ap50"]["D10"], 4),
                "ap_D20": round(metrics["per_class_ap50"]["D20"], 4),
                "ap_D40": round(metrics["per_class_ap50"]["D40"], 4),
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "f1": round(metrics["f1"], 4),
                "param_count": params["param_count"],
                "trainable_param_count": params["trainable_param_count"],
                "eval_runtime_sec": round(runtime, 1),
                "weights_path": str(weights.relative_to(REPO_ROOT)),
                "notes": ("PIPELINE SMOKE TEST - not a reportable result" if args.smoke
                         else f"Phase 4 zero-shot eval ({d['kind']}) on {dataset_key}"
                              + (f" -- SEEDED RANDOM SUBSAMPLE of {sample_size} images "
                                 f"(seed={SAMPLE_SEED}), not the full set" if sample_size else "")),
            })

            # Qualitative failure examples (skip for source -- it's the
            # in-domain reference, not a generalization failure case).
            if not args.skip_failure_examples and d["kind"] == "zero-shot":
                worst = select_failure_examples(
                    image_paths, preds, gts, k=args.failure_examples_per_country)
                out_dir = FAILURE_DIR / cfg["model_name"] / dataset_key
                for item in worst:
                    # NOTE: named failure_img_path, NOT out_path -- out_path is
                    # the CSV path from the top of main(). This used to shadow
                    # it (same name, reassigned every loop iteration), which
                    # would silently redirect every later write_results_csv
                    # call in this run to whatever failure-example .jpg path
                    # happened to be last, instead of the actual results CSV.
                    failure_img_path = out_dir / item["image_path"].name
                    draw_gt_and_predictions(item["image_path"], item["gt"], item["pred"], failure_img_path)
                if worst:
                    print(f"  saved {len(worst)} failure examples to {out_dir}")

    rebuild_full_results_csv(phase, out_path)
    print(f"\nWrote {out_path}")

    if args.smoke:
        print("\nSMOKE TEST run -- these are pipeline-validation numbers, not results.")
        print("Rerun without --smoke for the real Phase 4 evaluation.")


if __name__ == "__main__":
    main()
