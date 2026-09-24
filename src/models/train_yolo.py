"""
Phase 3a: train a plain (unmodified) YOLOv8 on the SOURCE split.

No attention module -- this is the reference baseline that Phase 3b's
coordinate-attention variant is measured against (the project spec section 3).

CHUNKED TRAINING
----------------
Free Colab sessions rarely survive a full 100-epoch run, so --chunk-epochs N
trains at most N epochs per invocation and then stops cleanly. Rerunning the
same command resumes exactly where it left off.

This is a TRUE resume, not a warm restart: ultralytics restores the optimizer
state, EMA, AMP scaler, best-fitness, and the LR scheduler position
(trainer.scheduler.last_epoch = start_epoch - 1). Training in five 20-epoch
chunks is therefore equivalent to one 100-epoch run -- the LR schedule still
spans 100 epochs rather than restarting five times. A naive
`YOLO(last.pt).train(epochs=20)` would instead restart the schedule each chunk
and produce a different (worse) model, so it is deliberately NOT used here.

ONE SUBTLETY THIS CODE WORKS AROUND: when a run ends, ultralytics calls
final_eval() -> strip_optimizer(last.pt), which deletes the optimizer state and
sets epoch = -1. resume_training() then asserts 0 < start_epoch and fails, so a
cleanly-stopped run would be UNRESUMABLE. An on_model_save callback therefore
keeps an unstripped copy (weights/last_resumable.pt) written before the strip,
and that copy is restored before resuming.

Usage:
    # full run in one go
    python src/models/train_yolo.py --data config/dataset_source.yaml --name yolov8n_source

    # 20 epochs at a time -- rerun the SAME command to continue
    python src/models/train_yolo.py --data config/dataset_source.yaml \
        --name yolov8n_source --chunk-epochs 20 --device 0
"""

import argparse
import csv
import shutil
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "utils"))
sys.path.insert(0, str(REPO_ROOT / "src" / "eval"))

from seed import set_global_seed  # noqa: E402
from experiment_logger import log_run, config_hash  # noqa: E402

RUNS_DIR = REPO_ROOT / "experiments" / "runs"

# Unstripped checkpoint kept for resuming across chunks (see module docstring).
RESUMABLE_NAME = "last_resumable.pt"

# Every detection-relevant augmentation knob ultralytics exposes, zeroed out
# for the Phase 5 "augmentation off" ablation (the project spec section 5.4 item 2).
# Values taken from ultralytics/cfg/default.yaml; the baseline ("augment on")
# path never touches these and just uses ultralytics' own defaults, which is
# what Phase 3a actually trained with -- adding this dict changes nothing for
# any run that doesn't pass --no-augment.
NO_AUGMENT_OVERRIDES = {
    "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
    "degrees": 0.0, "translate": 0.0, "scale": 0.0, "shear": 0.0, "perspective": 0.0,
    "flipud": 0.0, "fliplr": 0.0, "bgr": 0.0,
    "mosaic": 0.0, "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0,
}


def load_hyperparams() -> dict:
    with open(REPO_ROOT / "config" / "hyperparams.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def training_progress(run_dir: Path) -> tuple[int, int | None]:
    """(epochs_completed, total_epochs_configured) for an existing run directory.

    Epochs completed is read from results.csv, which ultralytics appends to once
    per finished epoch -- a more reliable record than the checkpoint, which can
    be mid-write if a session was killed.
    """
    results_csv = run_dir / "results.csv"
    completed = 0
    if results_csv.exists():
        with open(results_csv, encoding="utf-8") as f:
            completed = max(0, sum(1 for _ in csv.reader(f)) - 1)  # minus header

    total = None
    args_yaml = run_dir / "args.yaml"
    if args_yaml.exists():
        with open(args_yaml, encoding="utf-8") as f:
            total = (yaml.safe_load(f) or {}).get("epochs")

    return completed, total


def register_chunk_callbacks(model, weights_dir: Path, chunk_epochs: int | None):
    """Keep a resumable checkpoint, and stop after `chunk_epochs` epochs.

    on_model_save fires immediately after save_model() writes last.pt and BEFORE
    final_eval() strips it, so the copy taken here still holds optimizer state.
    on_fit_epoch_end fires after that save and before the loop's `if self.stop:
    break`, so setting stop there ends the run on a fully-saved epoch boundary.
    """
    state = {"start_epoch": None, "done_this_run": 0}

    def _on_model_save(trainer):
        src = Path(trainer.last)
        if src.exists():
            weights_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, weights_dir / RESUMABLE_NAME)

    def _on_fit_epoch_end(trainer):
        if state["start_epoch"] is None:
            state["start_epoch"] = trainer.epoch
        state["done_this_run"] = trainer.epoch - state["start_epoch"] + 1
        if chunk_epochs and state["done_this_run"] >= chunk_epochs:
            trainer.stop = True

    model.add_callback("on_model_save", _on_model_save)
    model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)
    return state


def _describe_run(args, model_label: str) -> str:
    """Human-readable notes field for experiment_log.csv, distinguishing
    baseline / 3b / ablation runs so the CSV is self-explanatory without
    needing to cross-reference the exact CLI flags used."""
    if args.smoke:
        return "PIPELINE SMOKE TEST - not a reportable result"
    if args.attention:
        return "Phase 3b: YOLOv8n + Coordinate Attention (single insertion, end of backbone)"
    if args.phase_label == "5":
        bits = [f"model={model_label}"]
        if args.no_augment:
            bits.append("augmentation=OFF")
        bits.append(f"imgsz={args.imgsz if args.imgsz is not None else 640}")
        return "Phase 5 ablation: " + ", ".join(bits)
    return "Phase 3a plain YOLOv8 baseline"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO_ROOT / "config" / "dataset_source.yaml"))
    ap.add_argument("--name", default=None)
    ap.add_argument("--epochs", type=int, default=None, help="override hyperparams.yaml TOTAL epochs")
    ap.add_argument("--chunk-epochs", type=int, default=None,
                    help="train at most this many epochs per invocation, then stop cleanly")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--attention", action="store_true",
                    help="Phase 3b: train YOLOv8n + CoordinateAttention instead of plain YOLOv8n. "
                         "Every other setting (hyperparams, seed, splits) is identical to Phase 3a "
                         "by construction -- this flag is the only thing that changes.")
    ap.add_argument("--base-model", default=None,
                    help="Phase 5 backbone-size ablation: override hyperparams.yaml's yolo.model "
                         "(e.g. yolov8s.pt, yolov8m.pt). Mutually exclusive with --attention -- "
                         "the CA yaml (config/yolov8n_ca.yaml) is sized for yolov8n's width only.")
    ap.add_argument("--no-augment", action="store_true",
                    help="Phase 5 augmentation ablation: disable every detection augmentation "
                         "knob (mosaic, mixup, HSV, geometric transforms, flips). Default (flag "
                         "absent) leaves ultralytics' own defaults untouched, which is what the "
                         "Phase 3a baseline trained with.")
    ap.add_argument("--smoke", action="store_true",
                    help="mark this run as a pipeline test, not a reportable result")
    ap.add_argument("--phase-label", default=None,
                    help="override the logged 'phase' column (default: 3b if --attention "
                         "else 3a). run_ablation.py passes '5' for backbone/augmentation/"
                         "resolution ablation runs.")
    args = ap.parse_args()

    if args.attention and args.base_model:
        raise SystemExit("--attention and --base-model are mutually exclusive: "
                         "config/yolov8n_ca.yaml is sized specifically for yolov8n's width "
                         "(width_multiple=0.25); it is not valid for yolov8s/m/l/x.")

    # A forgotten --name with --attention/--base-model would default to the
    # SAME run directory as the plain baseline and resume-corrupt it with a
    # different architecture's checkpoint. Pick a distinct default instead.
    if args.name is None:
        if args.attention:
            args.name = "yolov8n_ca_source"
        elif args.base_model:
            args.name = f"{Path(args.base_model).stem}_source"
        else:
            args.name = "yolov8n_source"

    hp = load_hyperparams()
    ycfg = dict(hp["yolo"])
    if args.epochs is not None:
        ycfg["epochs"] = args.epochs
    if args.batch is not None:
        ycfg["batch"] = args.batch
    if args.imgsz is not None:
        ycfg["imgsz"] = args.imgsz
    if args.base_model is not None:
        ycfg["model"] = args.base_model
    # Keep ycfg in sync with --no-augment so config_hash(ycfg) actually
    # differs between the augment-on baseline and the augment-off ablation --
    # without this, NO_AUGMENT_OVERRIDES would change what ultralytics trains
    # with but not what gets hashed/logged, silently making the two runs look
    # like they used identical config.
    ycfg["augment"] = not args.no_augment

    seed = hp["seed"]
    set_global_seed(seed)

    from ultralytics import YOLO

    if args.attention:
        # Needed even on the RESUME path below: resuming loads a pickled model
        # object straight from a checkpoint (no re-parsing of the yaml), but
        # unpickling a CoordinateAttention instance still requires the class
        # to be importable, and register_ca_module() is what makes the name
        # resolvable if ultralytics' loader ever re-parses the architecture
        # (e.g. on a version where checkpoint loading falls back to yaml
        # reconstruction). Calling it unconditionally here removes any doubt,
        # and it's idempotent.
        from attention import register_ca_module

        register_ca_module()

    run_dir = RUNS_DIR / args.name
    weights_dir = run_dir / "weights"
    resumable = weights_dir / RESUMABLE_NAME
    last_pt = weights_dir / "last.pt"

    completed, total_from_run = training_progress(run_dir)
    total_epochs = total_from_run or ycfg["epochs"]

    if completed >= total_epochs and completed > 0:
        print(f"Training already complete: {completed}/{total_epochs} epochs. Skipping to evaluation.")
        train_runtime = 0.0
    else:
        if completed > 0 and resumable.exists():
            # Restore the unstripped checkpoint over last.pt so ultralytics'
            # resume machinery finds optimizer state and a valid epoch counter.
            shutil.copy2(resumable, last_pt)
            print(f"Resuming {args.name}: {completed}/{total_epochs} epochs done")
            model = YOLO(str(last_pt))
            train_kwargs = {"resume": True}
        else:
            if completed > 0 and not resumable.exists():
                raise SystemExit(
                    f"Run '{args.name}' has {completed} epochs in results.csv but no "
                    f"{RESUMABLE_NAME}. Its last.pt was stripped of optimizer state and "
                    f"cannot be resumed. Delete {run_dir} and start over, or use a new --name."
                )
            print(f"Starting {args.name}: 0/{total_epochs} epochs")
            if args.attention:
                from yolo_attention import build_ca_model

                model = build_ca_model(pretrained_weights=ycfg["model"], validate=True)
            else:
                model = YOLO(ycfg["model"])
            train_kwargs = {
                "data": args.data,
                "epochs": total_epochs,
                "imgsz": ycfg["imgsz"],
                "batch": ycfg["batch"],
                "workers": ycfg["workers"],
                "optimizer": ycfg["optimizer"],
                "patience": ycfg["patience"],
                "seed": seed,
                "deterministic": True,
                "project": str(RUNS_DIR),
                "name": args.name,
                "exist_ok": True,
            }
            if args.no_augment:
                train_kwargs.update(NO_AUGMENT_OVERRIDES)
                print("  augmentation: OFF (Phase 5 ablation)")

        if args.chunk_epochs:
            print(f"  chunk limit: {args.chunk_epochs} epochs this run")
        train_kwargs["device"] = args.device
        train_kwargs["verbose"] = True

        register_chunk_callbacks(model, weights_dir, args.chunk_epochs)

        # Epochs-done-this-run is read from results.csv (ground truth, same
        # source training_progress() uses for the real completion decision
        # below) rather than the chunk callback's own internal counter --
        # ultralytics can invoke on_fit_epoch_end an extra time around the
        # final revalidation pass, which would over-count a callback-based
        # tally by one. That extra call doesn't affect correctness (the stop
        # condition and results.csv are unaffected), only what a naive
        # counter would report, so ground truth is used for the print instead
        # of chasing the exact cause.
        completed_before, _ = training_progress(run_dir)

        t0 = time.perf_counter()
        model.train(**train_kwargs)
        train_runtime = time.perf_counter() - t0

        completed_after, _ = training_progress(run_dir)
        print(f"\nThis run trained {completed_after - completed_before} epochs "
              f"in {train_runtime / 60:.1f} min")

        completed, _ = training_progress(run_dir)

    # Only evaluate and log once ALL epochs are done. Logging a partially-trained
    # model would put a number in experiment_log.csv that looks like a Phase 3a
    # result but is not one.
    if completed < total_epochs:
        remaining = total_epochs - completed
        print("\n" + "=" * 64)
        print(f"  CHUNK COMPLETE -- {completed}/{total_epochs} epochs ({remaining} remaining)")
        print("=" * 64)
        print("  Not evaluated or logged yet: training is incomplete.")
        print("  Save your results to Drive, then rerun the SAME command to continue.")
        print("=" * 64 + "\n")
        return

    weights = weights_dir / "best.pt"
    if not weights.exists():
        raise SystemExit(f"Expected best weights at {weights}, not found")

    with open(args.data, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    test_images = Path(data_cfg["test"]) if "test" in data_cfg else Path(data_cfg["val"])

    from evaluate import run_evaluation, print_results, labels_dir_for

    test_labels = labels_dir_for(test_images)

    model_label = "yolov8n_ca" if args.attention else Path(ycfg["model"]).stem
    results = run_evaluation("yolo", weights, test_images, test_labels, hp,
                             device=args.device)
    print_results(f"YOLOv8 ({model_label}) - SOURCE test", results)

    phase_prefix = args.phase_label or ("3b" if args.attention else "3a")
    # config_hash is computed from ycfg the SAME way for every YOLO run
    # (architecture/imgsz/augmentation differences show up in ycfg itself, so
    # an ablation variant's hash correctly differs from the 3a baseline's --
    # unlike 3a vs 3b, where ycfg is identical and the architecture difference
    # lives entirely in "model" / the yaml, by design).
    log_run({
        "run_id": args.name,
        "phase": f"{phase_prefix}-smoke" if args.smoke else phase_prefix,
        "model": model_label,
        "config_hash": config_hash(ycfg),
        "dataset": "source_subset" if args.smoke else "source",
        "split": "test",
        "seed": seed,
        "epochs": total_epochs,
        "imgsz": ycfg["imgsz"],
        "batch": ycfg["batch"],
        "num_images": results["num_images"],
        "map50": round(results["map50"], 4),
        "map50_95": round(results["map50_95"], 4),
        "ap_D00": round(results["per_class_ap50"]["D00"], 4),
        "ap_D10": round(results["per_class_ap50"]["D10"], 4),
        "ap_D20": round(results["per_class_ap50"]["D20"], 4),
        "ap_D40": round(results["per_class_ap50"]["D40"], 4),
        "precision": round(results["precision"], 4),
        "recall": round(results["recall"], 4),
        "f1": round(results["f1"], 4),
        "latency_ms": results["latency_ms"],
        "fps": results["fps"],
        "param_count": results["param_count"],
        "trainable_param_count": results["trainable_param_count"],
        "train_runtime_sec": round(train_runtime, 1),
        "eval_runtime_sec": results["eval_runtime_sec"],
        "weights_path": str(weights.relative_to(REPO_ROOT)),
        "notes": _describe_run(args, model_label),
    })
    print("Logged run to experiments/results/experiment_log.csv")


if __name__ == "__main__":
    main()
