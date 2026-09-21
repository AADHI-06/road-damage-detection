# CLAUDE.md — Road Damage Detection Project

> Project memory for Claude Code. Read this fully before writing any code.
> **Version 2** — updated with DA-RDD as primary base paper and Phase 3a/3b split.
> **Version 3** — added Phase 7: a team-added Tier 1 demo dashboard (section 2.1),
> built only once the final model is chosen. Not part of the official rubric (§11).
> **Version 4** — added YOLO26n as a second one-stage comparison model (team
> addition, trained after Phase 3a with the same SOURCE split / seed 42, at
> batch 8; see §3). Does not change the Faster-R-CNN-vs-YOLOv8 headline framing —
> it is a third data point on the same axis, not a new paradigm.

---

## 0. Working Agreement (read first, applies to every task)

**This is a graded academic project with a viva.** The student must be able to explain every file in this repo under questioning. Therefore:

1. **Work one phase at a time.** Do not implement multiple phases in a single session. Stop at each phase boundary and let the student review.
2. **Explain before you build.** Before writing a non-trivial script, state in 3–5 lines what it will do and why. Wait for confirmation on anything architectural.
3. **Never fabricate results.** Do not write placeholder metrics, example mAP values, or "expected" numbers into reports, README files, or result tables. Every number in this repo must come from an actual executed run. If a run hasn't happened, write `NOT YET RUN`.
4. **Prefer readable over clever.** Straightforward code the student can defend beats optimized code they can't.
5. **Comment the non-obvious.** Especially coordinate conversions, split logic, and metric definitions — these are the most likely viva questions.
6. **Log every experiment.** Every run appends to `experiments/results/experiment_log.csv`. No silent runs.
7. **Flag assumptions.** If dataset structure, class mapping, or a path differs from what's documented here, stop and report it rather than guessing around it.

---

## 1. Project Identity

**Title:** Multi-Class Road Damage Detection and Cross-Regional Generalization Analysis Using Deep Neural Networks

**Course:** BCSE332L — Deep Learning (Course-Based Design Project), VIT Vellore

**Team:** 2 members (Student A, Student B — fill in real names in README)

**Core research question:** Does a road damage detector trained on one set of countries generalize to unseen countries, by how much does performance degrade per country, and does detector architecture affect that degradation?

**Why it matters:** Automated road inspection is only deployable if a model trained in one region works elsewhere. The base paper (DA-RDD) assumes this gap exists and jumps to fixing it via adversarial adaptation; this project **quantifies the gap per-country first**, which is the diagnostic step DA-RDD does not report.

---

## 2. Scope Boundaries (do not drift)

**IN scope:**
- Multi-class damage detection on RDD2022 (D00, D10, D20, D40)
- Two architecturally distinct detectors: Faster R-CNN (two-stage) and YOLOv8 (one-stage)
- YOLO26n — a second one-stage model, trained as an extra comparison point (team
  addition, §3). Same SOURCE split, seed 42, 100 epochs, 640px, batch 8. Not a
  new paradigm, not "more advanced than" YOLOv8 — just another one-stage reading.
- Cross-country zero-shot generalization experiment (the headline result)
- Hyperparameter tuning + ablation studies
- Inference speed benchmarking
- **Tier 1 demo dashboard** (team-added deliverable, NOT part of the official 30-mark
  rubric in §11 — see §2.1): a hosted, single-page app where a user uploads one image
  and gets back detections drawn on it. Build only after the final model is chosen
  (post Phase 4 review, post Phase 3b decision) — see Phase 7 in §7.

**OUT of scope (do not build unless explicitly asked):**
- Rutting / patchwork detection — **not labeled in RDD2022**, do not invent these classes
- Adversarial domain adaptation (DA-RDD's actual method) — future work, NOT a deliverable. Adversarial training is unstable and can silently fail to converge. Mention in the report's future-work section only.
- Anything beyond the Tier 1 dashboard defined in §2.1 — no auth, no multi-user
  history, no database, no batch upload, no mobile app, no deployment pipeline
  beyond a single hosted instance
- Real-time video pipelines
- Any new dataset collection

**DEFERRED (Phase 3b — plan for it, don't build it yet):**
- Coordinate attention integrated into YOLOv8, with a with/without ablation

### 2.1 Tier 1 Demo Dashboard — exact scope

**What it does:** user uploads one image → the FINAL chosen model (see Phase 7) runs
inference → the image is returned with boxes, class labels (D00/D10/D20/D40), and
confidence scores drawn on it → a small table lists each detection (class, confidence,
box) and a per-image summary (count of damages by type, inference latency).

**What it deliberately does NOT do, and why:** it does not compute or display mAP,
F1, precision, or recall for the uploaded image. Those metrics require ground truth,
which a live upload never has — a dashboard cannot produce a real accuracy number for
an arbitrary photo, only a fabricated-looking one. Instead, the dashboard shows the
model's **already-measured** offline numbers as context (e.g. "this model: mAP@0.5 X
in-domain, Y on Norway zero-shot" pulled from `experiment_log.csv` /
`cross_country_results.csv`) — real numbers from Phase 3a/4, not live ones. Never
invent a per-image accuracy metric to make the UI look more complete (rule 3, working
agreement: never fabricate results).

**Which model:** YOLOv8n (whichever `best.pt` Phase 4/3b review lands on as final,
plain or CA-augmented) — not Faster R-CNN. 6MB vs. 159MB, ~17ms vs. ~60ms+ inference;
a demo needs to feel responsive, and hosting a 159MB checkpoint on a free tier is a
real, avoidable headache.

**Hosting:** a single free-tier hosted instance (e.g. HuggingFace Spaces, Streamlit
Community Cloud) is sufficient. No custom infra, no scaling concerns.

---

## 3. Model Plan — three tiers, built in two stages

**Important framing (this is a viva question):** YOLOv8 is NOT "more advanced" than Faster R-CNN. One-stage detectors are faster; two-stage detectors give better localization and recognition accuracy. They are different paradigms, not a linear upgrade path. Never present one as an upgrade of the other.

| Tier | Model | Role | Justification |
|---|---|---|---|
| Reference baseline | Faster R-CNN (ResNet-50 FPN) | Reproduces DA-RDD's detector backbone without its adaptation machinery | The base paper's own architecture (DA-RDD uses RPN + RoI losses) |
| Comparison baseline | YOLOv8 | Different detection paradigm (one-stage) | CRDDC-2022's winning solutions were YOLO-family; represents practical SOTA on this exact dataset |
| Comparison baseline (team addition) | YOLO26n | A newer one-stage detector, same paradigm as YOLOv8n | Adds a second point on the one-stage side so "which one-stage model degrades least cross-country" is answerable, not just one-stage-vs-two-stage. Trained after Phase 3a, identical SOURCE split / seed / schedule (batch 8, the only knob changed, for T4 memory headroom). |
| Proposed model (Phase 3b) | YOLOv8 + Coordinate Attention | The student's own contribution | Directional crack classes benefit from axis-wise positional encoding; CA-on-YOLO validated on RDD data by a CRDDC-2022 entry |

**Build order — do NOT merge these:**

### Phase 3a (build now)
Faster R-CNN + YOLOv8, both plain and unmodified. Train on SOURCE, evaluate in-domain. **Stop here for student review.** The student will look at the numbers before deciding whether to proceed to 3b.

### Phase 3b (build only when explicitly instructed)
Integrate coordinate attention into YOLOv8. Becomes the novel-component ablation.

**Why coordinate attention and not CBAM** (this is the viva answer — do not substitute CBAM): the three crack classes (D00 longitudinal, D10 transverse, D20 alligator) are directional, elongated structures. Coordinate attention factorizes pooling along the H and W axes separately, preserving axis-wise positional information. CBAM's spatial attention collapses channels into a single 2D map and discards that directional structure.

**Integration constraints for 3b:**
- Register the CA module in the Ultralytics model parser and reference it from a custom model YAML. Import alone will not work and fails silently.
- Insert CA at ONE location only: end of backbone, before the neck. Multiple insertions make the ablation uninterpretable.
- Identical hyperparameters, seed (42), and splits as 3a. Any other change invalidates the comparison.

**Keep the codebase 3b-ready during 3a:** leave `src/models/attention.py` and `src/models/yolo_attention.py` as stub files with a TODO comment. Do not implement them. Do not delete the hooks either.

---

## 4. Dataset — RDD2022 (already downloaded)

### 4.1 Expected structure
RDD2022 ships as per-country folders. Typical layout:

```
RDD2022/
├── China_Drone/train/{images/, annotations/xmls/}
├── China_MotorBike/train/{images/, annotations/xmls/}
├── Czech/train/{images/, annotations/xmls/}
├── India/train/{images/, annotations/xmls/}
├── Japan/train/{images/, annotations/xmls/}
├── Norway/train/{images/, annotations/xmls/}
└── United_States/train/{images/, annotations/xmls/}
```

**FIRST TASK, before anything else:** run a verification script that walks the actual downloaded directory and prints the real structure, per-country image counts, and annotation counts. **Do not assume the layout above is correct.** Report what's actually there.

### 4.2 Classes (use exactly these four)

| Code | Damage type |
|------|-------------|
| D00 | Longitudinal crack |
| D10 | Transverse crack |
| D20 | Alligator crack |
| D40 | Pothole |

Other codes (D43, D44, D50, etc.) appear in some country subsets. **Default: filter them out** and train on the four standard challenge classes, so results are comparable to CRDDC-2022 leaderboard work. Log how many boxes were dropped by this filter — that number goes in the report.

### 4.3 Known gotchas (these are real, expect them)

- **Annotations are Pascal VOC XML**, not YOLO format. Conversion is required and is the single most error-prone step in this project.
- **Image sizes differ by country.** Most countries use small square images (~600×600); **Norway uses high-resolution rectangular images**. This is not a bug — it's part of the domain shift being studied, and should be discussed in the report. Do not silently resize everything without recording it.
- **China appears as two subsets** (China_Drone, China_MotorBike) with different capture modalities. Drone imagery is a genuinely different domain from vehicle-mounted — treat and report them separately, do not merge blindly.
- **Norway's images have vehicles masked out.** Documented in the RDD2022 data article. Note it as a domain-shift factor.
- **Class imbalance is significant** — potholes (D40) are far rarer than cracks. Record per-class counts before training. The survey (Fan et al. 2024, §V.A) explicitly names class imbalance as a key open issue in this field — cite it when discussing this.
- Some XML files may reference images that don't exist, or vice versa. The verification script must catch and report orphans rather than crashing mid-conversion.
- RDD2022's public release provides labeled **train** data; the official CRDDC test set is held out. **Build your own test split from the labeled data** — never claim to evaluate on the official challenge test set.

---

## 5. Experimental Design (the core of the project)

### 5.1 Country split — fixed, do not change without noting it

- **SOURCE (train + val + in-domain test):** India + Japan
- **TARGET (zero-shot, never seen during training):** Czech, Norway, United States, China

Within SOURCE: 70% train / 15% val / 15% in-domain test, split at image level with a fixed random seed (`seed=42`) for reproducibility.

### 5.2 The headline result
`in-domain mAP` (India+Japan test) **vs.** `zero-shot mAP` (each TARGET country separately), **for each architecture**.

Three comparisons come out of this:
1. Faster R-CNN vs. YOLOv8 → paradigm trade-off (accuracy vs. speed)
2. In-domain vs. cross-country, per model → the generalization gap
3. **Which architecture degrades least under domain shift** → the headline finding, and the question DA-RDD doesn't answer because it jumps straight to adaptation

YOLO26n (the team-added third model, §3) rides comparisons 2 and 3 as a second
one-stage line — it sharpens comparison 3 into "does a *newer* one-stage design
generalize any better than YOLOv8n," without touching the one-stage-vs-two-stage
framing of comparison 1.

Report **per-country**, never averaged away. The variation between countries is itself a result.

### 5.3 Metrics to compute (all of them, every run)
- mAP@0.5 and mAP@0.5:0.95
- Per-class AP (D00, D10, D20, D40)
- Precision, Recall, **F1** (F1 is what CRDDC-2022 used — needed for leaderboard comparison)
- Inference latency (ms/image) and throughput (FPS), measured on identical hardware for both models
- Parameter count
- Per-country breakdown for every metric above

External comparison point (cite, do not reproduce as your own): the top CRDDC'2022 model reached F1 ≈ 76.9% across all six countries on RDD2022.

### 5.4 Ablations (Phase 5)
1. Backbone size: YOLOv8n vs. YOLOv8s vs. YOLOv8m
2. Data augmentation: on vs. off
3. Input resolution: 416 vs. 640 vs. 960
4. **Coordinate attention: with vs. without** (Phase 3b only)

Each ablation changes **one variable at a time**. Keep everything else fixed. Log all of it.

---

## 6. Repository Structure

```
road-damage-detection/
├── CLAUDE.md                       # this file
├── README.md                       # student-facing summary, updated as work progresses
├── requirements.txt
├── .gitignore                      # must exclude data/, experiments/runs/, *.pt
│
├── config/
│   ├── dataset_source.yaml         # India+Japan YOLO data config
│   ├── dataset_target_czech.yaml   # one per held-out country
│   ├── dataset_target_norway.yaml
│   ├── dataset_target_us.yaml
│   ├── dataset_target_china.yaml
│   └── hyperparams.yaml            # single source of truth for training hyperparameters
│
├── data/
│   ├── raw/RDD2022/                # downloaded dataset — READ ONLY, never modify
│   ├── processed/
│   │   ├── source/                 # India+Japan, YOLO format
│   │   │   ├── images/{train,val,test}/
│   │   │   └── labels/{train,val,test}/
│   │   └── target/                 # per-country zero-shot eval sets
│   │       ├── czech/{images,labels}/
│   │       ├── norway/{images,labels}/
│   │       ├── us/{images,labels}/
│   │       └── china/{images,labels}/
│   └── DATA_REPORT.md              # auto-generated: counts, class distribution, dropped boxes
│
├── src/
│   ├── data/
│   │   ├── verify_dataset.py       # PHASE 1 — run this first
│   │   ├── voc_to_yolo.py          # XML → YOLO txt conversion
│   │   ├── build_splits.py         # source/target split construction
│   │   └── dataset_stats.py        # class counts, image size distribution, plots
│   ├── models/
│   │   ├── train_faster_rcnn.py    # PHASE 3a
│   │   ├── train_yolo.py           # PHASE 3a
│   │   ├── attention.py            # PHASE 3b — STUB ONLY until instructed
│   │   └── yolo_attention.py       # PHASE 3b — STUB ONLY until instructed
│   ├── eval/
│   │   ├── evaluate.py             # single-model, single-dataset evaluation
│   │   ├── cross_country_eval.py   # THE core experiment
│   │   ├── metrics.py              # mAP/F1/precision/recall implementations
│   │   └── benchmark_speed.py      # latency + FPS + param count
│   ├── ablation/
│   │   └── run_ablation.py
│   └── utils/
│       ├── experiment_logger.py    # appends to experiment_log.csv
│       ├── seed.py                 # global seeding for reproducibility
│       └── visualize.py            # prediction overlays, confusion matrices, result plots
│
├── experiments/
│   ├── runs/                       # model weights + training curves (gitignored)
│   └── results/
│       ├── experiment_log.csv      # every run, every config, every metric
│       ├── cross_country_results.csv
│       ├── ablation_results.csv
│       └── figures/
│
├── notebooks/
│   └── 01_dataset_exploration.ipynb
│
├── reports/
│   ├── phase1_problem_and_literature.md
│   ├── phase2_model_design.md
│   ├── final_report.md             # HARD LIMIT: 10 pages
│   └── figures/
│
├── demo/                           # PHASE 7 — build only once the final model is chosen
│   ├── app.py                      # Tier 1 Gradio/Streamlit dashboard (see section 2.1)
│   ├── requirements.txt            # demo-only deps (gradio/streamlit, ultralytics) —
│   │                                # kept separate from the root requirements.txt so
│   │                                # hosting platforms don't need the training stack
│   └── README.md                   # how to run locally + the hosted URL once deployed
│
└── scripts/
    ├── 00_verify_data.sh
    ├── 01_prepare_data.sh
    ├── 02_train_baselines.sh
    ├── 03_cross_country_eval.sh
    └── 04_run_ablations.sh
```

---

## 7. Execution Phases

Work through these **in order**. Stop after each for student review.

### Phase 1 — Verify & Understand the Data
- `verify_dataset.py`: walk `data/raw/RDD2022/`, report actual structure, per-country image/annotation counts, orphaned files, XML parse failures.
- `dataset_stats.py`: per-class box counts per country, image size distribution, plots.
- Output `data/DATA_REPORT.md`.

### Phase 2 — Data Pipeline
- `voc_to_yolo.py`: VOC XML → YOLO normalized format. Handle the four target classes, filter and count others, validate all converted coordinates fall in [0,1].
- `build_splits.py`: construct SOURCE (India+Japan, 70/15/15) and TARGET (per-country) sets. Seed-fixed.
- **Mandatory sanity check:** render ~20 converted labels back onto their images (include ≥5 India, ≥5 Japan, some D40 potholes, and ≥1 Norway sample since its resolution differs). Save to `experiments/results/figures/label_check/`. **This catches conversion bugs that silently destroy training. Do not skip it.**

### Phase 3a — Baseline Models (CURRENT TARGET)
- Train Faster R-CNN (ResNet-50 FPN, via `torchvision`) on SOURCE train, validate on SOURCE val.
- Train YOLOv8 on the same split.
- Evaluate both on SOURCE test → in-domain numbers.
- Benchmark inference speed and parameter count for both.
- **STOP. Report numbers. Wait for student decision on Phase 3b.**

### Phase 3b — Coordinate Attention (ONLY when explicitly instructed)
- Implement CA module, integrate into YOLOv8 backbone, retrain with identical settings.
- Produces the with/without ablation.

### Phase 4 — Cross-Country Generalization (core experiment)
- `cross_country_eval.py`: load each trained model, evaluate zero-shot on each TARGET country separately.
- Produce the degradation table: model × country × metric.
- Generate qualitative failure examples — a few images per country where the model fails, for the report's analysis section.

### Phase 5 — Hyperparameter Tuning & Ablations
- Run the ablation grid from §5.4, one variable at a time.

### Phase 6 — Reporting Support
- Generate all final tables and figures from `experiments/results/`.
- Assist with report drafting — but the student writes the analysis, justification, and AI-pair-programming reflection sections themselves (viva-graded).

### Phase 7 — Demo Dashboard (team-added deliverable, see section 2.1)
- **Do not start before the final model is chosen** — i.e. after Phase 4 is reviewed and the Phase 3b decision is made, so the demo points at the actual final checkpoint instead of one that gets superseded.
- Build `demo/app.py`: single-image upload → inference with the final YOLOv8 checkpoint → boxes + class + confidence drawn on the image + a detections table + per-image summary (count by damage type, inference latency).
- Also surface the model's real, already-measured offline numbers (in-domain mAP, cross-country degradation for at least one target country) as static context — pulled from `experiment_log.csv` / `cross_country_results.csv`, never computed live (see section 2.1 for why a live accuracy number is not possible).
- Deploy to one free-tier host (HuggingFace Spaces or Streamlit Community Cloud). Record the URL in `demo/README.md` and the root `README.md`.

---

## 8. Technical Environment

- Python 3.10+
- PyTorch (CUDA if available; must also run on Colab free-tier T4)
- `ultralytics` (YOLOv8)
- `torchvision` (Faster R-CNN)
- `opencv-python`, `pandas`, `matplotlib`, `pyyaml`, `tqdm`, `lxml`

**Compute guidance:** start with YOLOv8n and a subset of the data to validate the full pipeline end-to-end before launching long runs. A pipeline bug found after a 6-hour training run is the most expensive failure mode in this project.

---

## 9. Reproducibility Rules

- Global seed = 42, set for `random`, `numpy`, and `torch` via `src/utils/seed.py`.
- All hyperparameters live in `config/hyperparams.yaml` — never hardcode them in training scripts.
- Every run writes a row to `experiments/results/experiment_log.csv` with: timestamp, model, config hash, dataset split, all metrics, hardware, runtime.
- Model weights are gitignored; the log is not.

---

## 10. Literature Anchors (for the report; do not misattribute)

| Paper | Venue | Role |
|---|---|---|
| **C. Lin, D. Tian, X. Duan, J. Zhou, D. Zhao, D. Cao, "DA-RDD: Toward Domain Adaptive Road Damage Detection Across Different Countries" (2023)** | **IEEE Transactions on Intelligent Transportation Systems**, vol. 24, no. 3, pp. 3091–3103. DOI: 10.1109/TITS.2022.3221067 | **PRIMARY BASE PAPER.** A method paper on cross-country road damage detection, built on RDD2020 (this dataset's predecessor), using a Faster R-CNN backbone (RPN + RoI losses). Justifies both the problem framing and the Faster R-CNN reference baseline. |
| L. Fan et al., "Pavement Defect Detection With Deep Learning: A Comprehensive Survey" (2024) | IEEE Transactions on Intelligent Vehicles, vol. 9, no. 3, pp. 4292–4311. DOI: 10.1109/TIV.2023.3326136 | Supporting survey. **§V.C "Model Generalization Ability" explicitly names cross-domain generalization as an unsolved key issue** — cite as the literature-anchored research gap. §V.A covers small-sample/class-imbalance; §V.B covers accuracy vs. real-time trade-off (justifies speed benchmarking); §III.A.2 establishes the one-stage vs. two-stage taxonomy. |
| Arya et al., "RDD2022: A multi-national image dataset for automatic road damage detection" (2024) | Geoscience Data Journal | Dataset provenance. |
| Arya et al., "Crowdsensing-based Road Damage Detection Challenge (CRDDC-2022)" | IEEE BigData 2022 (conference, **not** Transactions) | Empirical model justification: YOLO-family solutions dominated the leaderboard; top model reached F1 ≈ 76.9% across all six countries. |

### Critical distinction to state honestly in the report
DA-RDD performs **domain adaptation** — it uses unlabeled target-country data during training to align feature distributions adversarially. This project performs **zero-shot generalization measurement** — training on source countries and evaluating on target countries with no adaptation whatsoever.

Frame this as deliberate: DA-RDD assumes the gap exists and jumps to fixing it; this project quantifies the gap per-country first, the diagnostic step DA-RDD does not report. If asked "why not just implement DA-RDD," the answer is that adversarial domain-adaptation training is substantially harder to stabilize, and the per-country measurement is a distinct and necessary contribution.

**Do not blur this distinction.** Never imply the project implements DA-RDD's method.

---

## 11. Rubric Alignment (30 marks — keep this visible)

| Phase | Criterion | Marks | What this repo must produce |
|---|---|---|---|
| I | Problem Identification | 2 | `reports/phase1_*.md` |
| I | Literature Review & Comparative Analysis | 2 | Comparative table: one-stage vs. two-stage detectors, DA-RDD's adaptation approach, disparity-based methods |
| I | Research Gap Identification | 2 | Cross-country generalization gap — anchored in survey §V.C and DA-RDD's framing |
| I | Objectives & Justification | 2 | **4 objectives total — 2 per student, named** |
| I | Dataset Selection & Report Quality | 2 | `data/DATA_REPORT.md` |
| II | Model Architecture Design | 2 | Architecture diagram in `reports/figures/` (strengthened by Phase 3b if pursued) |
| II | Implementation of Multiple Models | 2 | Faster R-CNN + YOLOv8 + YOLO26n all trained |
| II | Hyperparameter Tuning | 2 | `ablation_results.csv` |
| II | Experimental Analysis | 2 | Cross-country results + analysis |
| II | Video Demo & AI Pair Programming Reflection | 2 | **Student-written after building — do not pre-draft** |
| Final | Comparative Analysis | 2 | Model × country comparison tables |
| Final | Error Metrics & Performance Evaluation | 2 | Full metric suite, per-country |
| Final | Justification of Proposed Model | 2 | **Weakest criterion with only two off-the-shelf models. Phase 3b directly strengthens this.** |
| Final | Technical Report Quality & Citations | 2 | `final_report.md`, **10 pages max**, individually cited |
| Final | Individual Viva, Reflection & Demonstration | 2 | **Student must understand all code — this is why phases are reviewed** |

---

## 12. Anti-Patterns (do not do these)

- ❌ Writing example/placeholder metric values anywhere
- ❌ Claiming evaluation on the official CRDDC-2022 test set (it's held out — you're using your own split)
- ❌ Implying the project implements DA-RDD's adversarial adaptation method
- ❌ Presenting YOLOv8 as "more advanced than" Faster R-CNN — they are different paradigms
- ❌ Adding rutting/patchwork classes that don't exist in the data
- ❌ Modifying anything inside `data/raw/`
- ❌ Skipping the label-visualization sanity check in Phase 2
- ❌ Building Phase 3b before Phase 3a numbers have been reviewed
- ❌ Running all phases in one session without student review
- ❌ Writing the AI-pair-programming reflection on the student's behalf
- ❌ Averaging away per-country results — the per-country variation *is* the finding
- ❌ Building the Phase 7 demo dashboard before the final model (post Phase 4/3b review) is chosen
- ❌ Displaying a live-computed accuracy metric (mAP/F1/precision/recall) for a user-uploaded image in the dashboard — no ground truth exists for it; show the model's real offline numbers as context instead (section 2.1)
