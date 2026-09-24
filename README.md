# Multi-Class Road Damage Detection and Cross-Regional Generalization Analysis

**Course:** BCSE332L — Deep Learning (Course-Based Design Project), VIT Vellore
**Team:** Aadhithya A (23BAI0048), Ashwin B (23BAI0077)

**Research question:** Does a road damage detector trained on one set of countries
generalize to unseen countries, and by how much does performance degrade?

---

## Status

| Phase | Description | Status |
|---|---|---|
| 1 | Verify & understand the data | Complete |
| 2 | Data pipeline (VOC → YOLO, splits) | Complete |
| 3a | Plain baselines (Faster R-CNN + YOLOv8 + YOLO26n) | **Complete** — trained on Colab T4, reviewed. YOLO26n added later as a second one-stage comparison model (same SOURCE split / seed 42 / 100 ep / 640px, batch 8). See `experiments/results/experiment_log.csv` |
| 3b | Coordinate attention variant | **Complete (in-domain)** — CA module (`src/models/attention.py`), custom architecture (`config/yolov8n_ca.yaml`) trained 100 epochs on SOURCE, identical hyperparameters/seed to plain YOLOv8n (+6,680 params). In-domain with/without ablation is in `reports/results_tables.md`; not yet re-run through Phase 4's cross-country eval |
| 4 | Cross-country generalization | **Complete** (with caveats) — YOLOv8n and YOLO26n full-scale on all shared target countries; Faster R-CNN on source+czech full-scale, norway/us/china_motorbike on a 1,500-image PRELIMINARY subsample (re-run at full scale before final report — see `reports/results_tables.md`); china_drone is YOLOv8n-only |
| 5 | Hyperparameter tuning & ablations | Scaffolded (`src/ablation/run_ablation.py`), pipeline-verified on a subset, not yet run at full scale |
| 6 | Reporting support | `src/eval/generate_report_tables.py` built — auto-generates `reports/results_tables.md` and figures from `experiments/results/`, rerun any time new results land |
| 7 | Demo dashboard (team-added, not in the official rubric — see [demo/README.md](demo/README.md)) | **Built and tested locally**, all four trained models selectable (Faster R-CNN, YOLOv8n, YOLOv8n+CA, YOLO26n), plus a compare-all view with EigenCAM explanations for the YOLO family. Not deployed to a public host — local run only |

**Phase 3a baselines are trained and reviewed.** In-domain (SOURCE test) results:
YOLOv8n mAP@0.5 0.494, F1 0.526; Faster R-CNN mAP@0.5 0.507, F1 0.399;
YOLO26n mAP@0.5 0.486, F1 0.525 (2.5M params, the lightest of the three). Full
per-class breakdown in `experiments/results/experiment_log.csv`. Full cross-country
and Phase 3a tables: [reports/results_tables.md](reports/results_tables.md).

---

## Demo

Runs locally only (see [demo/README.md](demo/README.md) for setup) — no public host,
by deliberate choice: GitHub + a local run is sufficient for a graded course project
with a viva, and it removes any dependency on a free-tier host staying up on demo day.
Upload one image, pick a model (Faster R-CNN, YOLOv8n, YOLOv8n+CA, or YOLO26n), get back
detections (class, confidence, boxes drawn on the image). A "compare all models" view
runs all four side by side, adding an EigenCAM attention heatmap and a textual
explanation for each YOLO-family detection. It does not display a live-computed accuracy
metric for the uploaded image — that needs ground truth a single upload never has.

---

## Dataset

RDD2022, seven country subsets, 38,385 labeled train images. Full breakdown
(per-country counts, class distribution, dropped boxes, image sizes) is in
[data/DATA_REPORT.md](data/DATA_REPORT.md).

- **SOURCE** (train/val/test, 70/15/15, seed 42): India + Japan — 18,212 images
- **TARGET** (zero-shot, never trained on): Czech, Norway, United States,
  China_Drone, China_MotorBike (kept separate — different capture modalities,
  see `data/DATA_REPORT.md` section 8)

Classes: `D00` longitudinal crack, `D10` transverse crack, `D20` alligator crack,
`D40` pothole. Non-standard codes (D43/D44/D50/Repair/…) are filtered out and counted.

---

## Compute requirement

Training requires a CUDA GPU. This project's development machine has an **AMD GPU
and therefore runs PyTorch on CPU only**. Measured on this machine:

| Model | Measured rate | Full SOURCE epoch | Full run |
|---|---|---|---|
| YOLOv8n | 0.39 s/img train | ~89 min | 2.5–6.2 days (40–100 ep) |
| Faster R-CNN R50-FPN | 3.03 s/img train | ~10.7 h | ~5.4 days (12 ep) |

Both figures are measured on this machine (AMD Ryzen 9 5980HX, CPU-only), not
estimated. Together that is **8–12 days** of continuous compute for Phase 3a.

Training therefore runs on Google Colab (free T4):

```bash
python scripts/make_colab_bundle.py     # -> dist/rdd_bundle.zip
```

Upload that zip to Google Drive, then run
[notebooks/phase3a_colab.ipynb](notebooks/phase3a_colab.ipynb), which verifies the
GPU, checks the upload against `data/split_report.json`, runs the unit tests, and
trains both baselines.

### Chunked training (for session-limited environments)

Both training scripts accept `--chunk-epochs N`: train at most N epochs, stop cleanly,
save. Rerunning the same command resumes from the last completed epoch. The notebook uses
this to split YOLOv8 into 5 chunks of 20 and Faster R-CNN into 4 chunks of 3.

This is a **true resume** — optimizer state, EMA, and LR-scheduler position are all
restored, so chunked training produces the same model as one uninterrupted run. (A naive
`YOLO(last.pt).train(epochs=20)` would restart the LR schedule each chunk and give a
different, worse model; that approach is deliberately not used.)

One non-obvious detail: when a YOLO run ends, ultralytics calls `strip_optimizer()` on
`last.pt`, which deletes the optimizer state and sets `epoch = -1`, making the checkpoint
unresumable. `train_yolo.py` therefore keeps an unstripped copy at
`weights/last_resumable.pt` via an `on_model_save` callback and restores it before
resuming.

---

## Setup

```bash
pip install -r requirements.txt
```

## Pipeline

```bash
bash scripts/00_verify_data.sh      # Phase 1: verify data, build DATA_REPORT.md
bash scripts/01_prepare_data.sh     # Phase 2: VOC -> YOLO, build splits, sanity check
DEVICE=0 bash scripts/02_train_baselines.sh   # Phase 3a: train both baselines
```

Before any long run, validate the pipeline end-to-end on a subset:

```bash
python src/data/make_subset.py --train 300 --val 100 --test 100
python src/models/train_yolo.py --data config/dataset_source_subset.yaml \
    --name yolov8n_smoke --epochs 2 --smoke
```

## Tests

```bash
python src/eval/test_metrics.py           # known-answer tests for mAP/P/R/F1
python src/models/test_rcnn_dataset.py    # YOLO -> torchvision conversion tests
```

---

## Notes on methodology

- Faster R-CNN and YOLOv8 are **different paradigms, not an upgrade path**: one-stage
  (YOLOv8) is faster, two-stage (Faster R-CNN) localises and recognises more accurately.
  Neither is presented as an upgrade of the other anywhere in this project.
- **YOLO26n** is a third trained model — a newer *one-stage* detector, added as a
  second point on the one-stage side (same SOURCE split, seed 42, schedule; only the
  batch size differs, 8 vs 16, for T4 memory headroom). It is not "more advanced than"
  YOLOv8n and does not change the one-stage-vs-two-stage framing — it lets the
  cross-country question be asked as "does a newer one-stage design generalize better."
- This project measures **zero-shot cross-country generalization**. It does **not**
  implement DA-RDD's adversarial domain adaptation, which uses unlabeled target data
  during training. Do not blur that distinction.

- **All three detectors are scored by the same metric code** (`src/eval/metrics.py`).
  Using each framework's built-in metrics would confound the model comparison
  with differences between metric implementations.
- Metric implementation is cross-checked against ultralytics' own validator:
  ground-truth counts match exactly and the dominant-class AP agrees to 4 decimals.
- Model selection for every baseline uses validation mAP@0.5, so the models
  differ only in architecture (and, for YOLO26n, batch size).
- Evaluation aborts loudly if zero ground-truth boxes load, rather than
  reporting a meaningless 0.0 for every metric.
- The demo dashboard (Phase 7) shows the model's real, already-measured offline
  metrics as static context — it never computes a live accuracy number for a
  user-uploaded image, since that needs ground truth which a single upload
  doesn't have.
