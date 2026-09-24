"""
Phase 3a: train a plain Faster R-CNN (ResNet-50 FPN) on the SOURCE split.

This is the two-stage counterpart to the single-stage YOLOv8 baseline
(the project spec section 2). torchvision has no built-in training loop for detection,
so one is written out explicitly here.

MODEL SELECTION: the best checkpoint is chosen by validation mAP@0.5, the same
criterion ultralytics uses for YOLOv8. Selecting on val loss instead would make
the two baselines differ in one more way than intended.

CHUNKED TRAINING: --chunk-epochs N trains at most N epochs per invocation, then
stops. Rerunning the same command resumes. last.pt carries the optimizer AND
scheduler state, so the LR decay continues across chunk boundaries instead of
restarting -- chunked training is equivalent to one uninterrupted run.

The dataloader is reseeded per epoch (seed + epoch) rather than once at startup,
so the shuffle order for epoch N is identical whether or not a chunk boundary
fell before it. Without that, resuming would draw a different batch order than
an uninterrupted run and the two would not be reproducible against each other.

Usage:
    python src/models/train_faster_rcnn.py --name faster_rcnn_source

    # 4 epochs at a time -- rerun the SAME command to continue
    python src/models/train_faster_rcnn.py --name faster_rcnn_source \
        --chunk-epochs 4 --device cuda

    # pipeline validation on the subset (results are NOT reportable)
    python src/models/train_faster_rcnn.py --name faster_rcnn_smoke \
        --data-root data/processed/source_subset --epochs 1 --smoke
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "utils"))
sys.path.insert(0, str(REPO_ROOT / "src" / "eval"))
sys.path.insert(0, str(REPO_ROOT / "src" / "models"))

from seed import set_global_seed  # noqa: E402
from experiment_logger import log_run, config_hash  # noqa: E402
from rcnn_dataset import YoloFormatDetectionDataset, collate_fn, LABEL_OFFSET  # noqa: E402
from rcnn_model import build_faster_rcnn  # noqa: E402
from metrics import evaluate_detections, CLASS_NAMES  # noqa: E402

RUNS_DIR = REPO_ROOT / "experiments" / "runs"


def load_hyperparams() -> dict:
    with open(REPO_ROOT / "config" / "hyperparams.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def validate_map(model, loader, device, max_images: int | None = None) -> dict:
    """Run the model over the val set and score it with the shared metric code."""
    model.eval()
    preds, gts = [], []
    seen = 0

    for images, targets in loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for out, tgt in zip(outputs, targets):
            preds.append({
                "boxes": out["boxes"].cpu().numpy().astype(np.float64),
                "scores": out["scores"].cpu().numpy().astype(np.float64),
                # Shift torchvision's 1..4 back to the project's canonical 0..3.
                "labels": out["labels"].cpu().numpy().astype(int) - LABEL_OFFSET,
            })
            gts.append({
                "boxes": tgt["boxes"].cpu().numpy().astype(np.float64),
                "labels": tgt["labels"].cpu().numpy().astype(int) - LABEL_OFFSET,
            })
        seen += len(images)
        if max_images is not None and seen >= max_images:
            break

    return evaluate_detections(preds, gts, num_classes=len(CLASS_NAMES))


def train_one_epoch(model, optimizer, loader, device, epoch: int, log_interval: int = 50):
    model.train()
    running = 0.0
    n_batches = 0

    for i, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # In train mode torchvision returns a dict of losses, not detections.
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running += float(loss.item())
        n_batches += 1
        if i % log_interval == 0:
            print(f"  epoch {epoch} batch {i}/{len(loader)}  loss={float(loss.item()):.4f}")

    return running / max(n_batches, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/processed/source",
                    help="dir containing images/{train,val,test} and labels/{train,val,test}")
    ap.add_argument("--name", default="faster_rcnn_source")
    ap.add_argument("--epochs", type=int, default=None, help="TOTAL epochs for the run")
    ap.add_argument("--chunk-epochs", type=int, default=None,
                    help="train at most this many epochs per invocation, then stop cleanly")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--val-max-images", type=int, default=None,
                    help="cap images used for the per-epoch val check (final test eval always uses all)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    hp = load_hyperparams()
    fcfg = dict(hp["faster_rcnn"])
    if args.epochs is not None:
        fcfg["epochs"] = args.epochs
    if args.batch_size is not None:
        fcfg["batch_size"] = args.batch_size

    seed = hp["seed"]
    set_global_seed(seed)
    device = torch.device(args.device)

    data_root = (REPO_ROOT / args.data_root) if not Path(args.data_root).is_absolute() else Path(args.data_root)
    out_dir = RUNS_DIR / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = YoloFormatDetectionDataset(
        data_root / "images" / "train", data_root / "labels" / "train", train=True, seed=seed)
    val_ds = YoloFormatDetectionDataset(
        data_root / "images" / "val", data_root / "labels" / "val", train=False, seed=seed)

    print(f"Training Faster R-CNN ({fcfg['backbone']}) on {data_root}")
    print(f"  train={len(train_ds)} val={len(val_ds)} epochs={fcfg['epochs']} "
          f"batch={fcfg['batch_size']} device={device} seed={seed}")

    # Generator seeded so shuffling order is reproducible across runs.
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=fcfg["batch_size"], shuffle=True,
                              num_workers=fcfg["workers"], collate_fn=collate_fn,
                              generator=generator)
    val_loader = DataLoader(val_ds, batch_size=fcfg["batch_size"], shuffle=False,
                            num_workers=fcfg["workers"], collate_fn=collate_fn)

    model = build_faster_rcnn(len(CLASS_NAMES) + 1, fcfg["imgsz"],
                              pretrained=fcfg["pretrained"]).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=fcfg["lr"], momentum=fcfg["momentum"],
                                weight_decay=fcfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=fcfg["lr_step_size"], gamma=fcfg["lr_gamma"])

    best_map = -1.0
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"
    total_epochs = fcfg["epochs"]
    start_epoch = 1

    # Resume from last.pt if this run was previously chunked. Optimizer AND
    # scheduler state are restored, so LR decay continues rather than restarting.
    if last_path.exists():
        ckpt = torch.load(str(last_path), map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        best_map = ckpt.get("best_map50", -1.0)
        start_epoch = ckpt["epoch"] + 1
        print(f"  resuming from epoch {start_epoch}/{total_epochs} "
              f"(best val mAP@0.5 so far {best_map:.4f})")

    if start_epoch > total_epochs:
        print(f"Training already complete: {total_epochs}/{total_epochs} epochs. Skipping to evaluation.")
        train_runtime = 0.0
    else:
        last_epoch_this_run = (min(total_epochs, start_epoch + args.chunk_epochs - 1)
                               if args.chunk_epochs else total_epochs)
        if args.chunk_epochs:
            print(f"  chunk limit: epochs {start_epoch}..{last_epoch_this_run} this run")

        t0 = time.perf_counter()
        for epoch in range(start_epoch, last_epoch_this_run + 1):
            epoch_t0 = time.perf_counter()

            # Reseed per epoch so the shuffle for epoch N is the same whether or
            # not a chunk boundary fell before it (see module docstring).
            generator.manual_seed(seed + epoch)

            mean_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
            scheduler.step()

            val_results = validate_map(model, val_loader, device, args.val_max_images)
            epoch_time = time.perf_counter() - epoch_t0
            print(f"epoch {epoch}/{total_epochs}  train_loss={mean_loss:.4f}  "
                  f"val_mAP@0.5={val_results['map50']:.4f}  ({epoch_time / 60:.1f} min)")

            if val_results["map50"] > best_map:
                best_map = val_results["map50"]
                torch.save({"model": model.state_dict(), "epoch": epoch,
                            "val_map50": best_map, "config": fcfg}, best_path)
                print(f"  new best val mAP@0.5={best_map:.4f} -> saved {best_path.name}")

            # Written every epoch so an interrupted session loses at most one epoch.
            torch.save({"model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch": epoch, "best_map50": best_map,
                        "config": fcfg}, last_path)

        train_runtime = time.perf_counter() - t0
        completed = last_epoch_this_run
        print(f"\nThis run trained {completed - start_epoch + 1} epochs in "
              f"{train_runtime / 60:.1f} min, best val mAP@0.5={best_map:.4f}")

        # Only evaluate and log when ALL epochs are done -- logging a partially
        # trained model would put a non-result into experiment_log.csv.
        if completed < total_epochs:
            print("\n" + "=" * 64)
            print(f"  CHUNK COMPLETE -- {completed}/{total_epochs} epochs "
                  f"({total_epochs - completed} remaining)")
            print("=" * 64)
            print("  Not evaluated or logged yet: training is incomplete.")
            print("  Save results to Drive, then rerun the SAME command to continue.")
            print("=" * 64 + "\n")
            return

    # Final in-domain evaluation on the held-out test split.
    from evaluate import run_evaluation, print_results

    results = run_evaluation(
        "faster_rcnn", best_path,
        data_root / "images" / "test", data_root / "labels" / "test", hp,
        device=args.device)
    print_results("Faster R-CNN (plain) - SOURCE test", results)

    log_run({
        "run_id": args.name,
        "phase": "3a-smoke" if args.smoke else "3a",
        "model": f"faster_rcnn_{fcfg['backbone']}",
        "config_hash": config_hash(fcfg),
        "dataset": "source_subset" if args.smoke else "source",
        "split": "test",
        "seed": seed,
        "epochs": total_epochs,
        "imgsz": fcfg["imgsz"],
        "batch": fcfg["batch_size"],
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
        "weights_path": str(best_path.relative_to(REPO_ROOT)),
        "notes": "PIPELINE SMOKE TEST - not a reportable result" if args.smoke
                 else "Phase 3a plain Faster R-CNN baseline",
    })
    print("Logged run to experiments/results/experiment_log.csv")


if __name__ == "__main__":
    main()
