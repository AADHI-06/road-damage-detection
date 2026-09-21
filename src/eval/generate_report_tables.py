"""
Phase 6: generate final tables and figures from experiments/results/
(CLAUDE.md section 7, Phase 6: "Generate all final tables and figures from
experiments/results/").

Deliberately SAFE TO RERUN AT ANY TIME, including with incomplete data --
this is the "integrate" step: once new rows land in experiment_log.csv or
cross_country_results.csv (e.g. after a Colab run finishes and its results
are merged back), rerunning this script picks them up automatically. Missing
(model, dataset) pairs are shown as PENDING, never silently omitted or
fabricated (CLAUDE.md working agreement rule 3).

Does NOT write any analysis prose -- only tables and charts. The student
writes the analysis, justification, and interpretation themselves
(CLAUDE.md section 7, Phase 6 and section 12's anti-pattern list).

Usage:
    python src/eval/generate_report_tables.py

Writes:
    reports/results_tables.md
    reports/figures/phase3a_comparison.png
    reports/figures/cross_country_map50.png
"""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "experiments" / "results"
REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

LOG_PATH = RESULTS_DIR / "experiment_log.csv"
CROSS_COUNTRY_PATH = RESULTS_DIR / "cross_country_results.csv"
ABLATION_PATH = RESULTS_DIR / "ablation_results.csv"

MODEL_LABELS = {"yolov8n": "YOLOv8n", "faster_rcnn_resnet50_fpn": "Faster R-CNN",
               "yolov8n_ca": "YOLOv8n + CA", "yolo26n": "YOLO26n"}
# Target countries in a fixed display order (source first, then alphabetical
# by the order they were added to the project -- matches cross_country_eval.py).
COUNTRY_ORDER = ["source", "czech", "norway", "us", "china_drone", "china_motorbike"]
COUNTRY_LABELS = {
    "source": "Source (in-domain)", "czech": "Czech", "norway": "Norway",
    "us": "United States", "china_drone": "China (Drone)",
    "china_motorbike": "China (MotorBike)",
}
# Countries where a known caveat applies -- printed as a footnote wherever
# they appear, so the caveat travels with the number rather than living only
# in a script docstring the reader may never see.
CAVEATS = {
    "norway": "PRELIMINARY. Faster R-CNN: 1,500-image seeded random subsample, not "
             "the full 8,161 -- a time-boxed reduction, not a final result "
             "(see src/eval/cross_country_eval.py). YOLOv8n and YOLO26n: full set.",
    "us": "PRELIMINARY. Faster R-CNN: 1,500-image seeded random subsample, not the "
         "full 4,805 -- time-boxed, re-run at full scale before reporting. "
         "YOLOv8n and YOLO26n: full set.",
    "china_motorbike": "PRELIMINARY. Faster R-CNN: 1,500-image seeded random "
                       "subsample -- most of the full 1,977, so less sampling "
                       "noise than Norway/US here -- time-boxed, re-run at full "
                       "scale before reporting. YOLOv8n and YOLO26n: full set.",
    "china_drone": "Evaluated for YOLOv8n only (before this country was dropped). "
                   "Not run for Faster R-CNN or YOLO26n (see src/eval/cross_country_eval.py "
                   "module docstring) -- not a like-for-like comparison for this country.",
}


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def latest_per_key(rows: list[dict], key_fn) -> dict:
    """Most recent row per key(row), by timestamp if present else insertion order."""
    latest = {}
    for i, r in enumerate(rows):
        k = key_fn(r)
        ts = r.get("timestamp", str(i))
        if k not in latest or ts >= latest[k][0]:
            latest[k] = (ts, r)
    return {k: v[1] for k, v in latest.items()}


def fnum(row: dict, key: str, digits: int = 4):
    """Format a numeric CSV field, or PENDING if missing/blank."""
    val = row.get(key, "")
    if val in ("", None):
        return "PENDING"
    try:
        return f"{float(val):.{digits}f}"
    except ValueError:
        return str(val)


def build_phase3a_table(log_rows: list[dict]) -> tuple[str, dict]:
    """Phase 3a in-domain baseline table. Returns (markdown, {model: row}) for charting."""
    phase3a = [r for r in log_rows if r["phase"] == "3a"]
    latest = latest_per_key(phase3a, lambda r: r["model"])

    lines = [
        "## Phase 3a — In-Domain Baseline (SOURCE test)",
        "",
        "> No single \"accuracy\" number exists for object detection. The score row "
        "to quote is **mAP@0.5** (primary) or **F1** (leaderboard-comparable, IoU "
        "0.5 / conf 0.25). Precision and Recall are the same-threshold components "
        "of F1.",
        "",
        "| Metric | " + " | ".join(MODEL_LABELS.get(m, m) for m in latest) + " |",
        "|---" * (len(latest) + 1) + "|",
    ]
    metrics = [("mAP@0.5", "map50"), ("mAP@0.5:0.95", "map50_95"),
              ("Precision", "precision"), ("Recall", "recall"), ("F1", "f1")]
    for label, key in metrics:
        lines.append(f"| {label} | " + " | ".join(fnum(latest[m], key) for m in latest) + " |")
    lines.append("| Parameters | " + " | ".join(
        f"{int(float(latest[m]['param_count'])):,}" if latest[m].get("param_count") else "PENDING"
        for m in latest) + " |")

    if not latest:
        lines.append("\n**No Phase 3a results logged yet.**")

    return "\n".join(lines), latest


def build_cross_country_table(cc_rows: list[dict]) -> tuple[str, dict]:
    """Model x country cross-country tables. Returns (markdown, data) for charting.

    Object detection has no single "accuracy" number (there are no true
    negatives to count), so this reports the two standard scores side by side:
    mAP@0.5 (the primary detection score) and F1 (the CRDDC-2022
    leaderboard-comparable score, at IoU 0.5 / conf 0.25 -- see
    config/hyperparams.yaml). Per-class AP and mAP@0.5:0.95 are in
    experiments/results/cross_country_results.csv.
    """
    by_model_country = {}
    for r in cc_rows:
        by_model_country.setdefault(r["model"], {})[r["dataset"]] = r

    models = list(by_model_country.keys())
    if not models:
        return "## Phase 4 — Cross-Country Generalization\n\n**No results logged yet.**", {}

    countries_present = [c for c in COUNTRY_ORDER
                         if any(c in by_model_country[m] for m in models)]

    def metric_table(metric_key: str) -> list[str]:
        rows = ["| Country | " + " | ".join(MODEL_LABELS.get(m, m) for m in models) + " |",
                "|---" * (len(models) + 1) + "|"]
        for c in countries_present:
            cells = []
            for m in models:
                r = by_model_country[m].get(c)
                cells.append(fnum(r, metric_key) if r else "PENDING")
            rows.append(f"| {COUNTRY_LABELS.get(c, c)} | " + " | ".join(cells) + " |")
        return rows

    lines = ["## Phase 4 — Cross-Country Generalization (zero-shot)", "",
             "> Detection has no single \"accuracy\" figure. **mAP@0.5** is the primary "
             "score; **F1** (IoU 0.5, conf 0.25) is the CRDDC-2022 "
             "leaderboard-comparable score. Both are reported per country below.",
             "",
             "### mAP@0.5", ""]
    lines += metric_table("map50")
    lines += ["", "### F1 score", ""]
    lines += metric_table("f1")

    footnotes = [f"- **{COUNTRY_LABELS.get(c, c)}:** {CAVEATS[c]}"
                for c in countries_present if c in CAVEATS]
    if footnotes:
        lines.append("")
        lines.append("**Caveats (state these wherever these tables are used):**")
        lines.extend(footnotes)

    return "\n".join(lines), by_model_country


def build_ablation_table() -> str:
    rows = read_csv_rows(ABLATION_PATH)
    if not rows:
        return ("## Phase 5 — Ablations\n\n"
                "**Not yet run at full scale.** `src/ablation/run_ablation.py` is built and "
                "pipeline-verified on a subset; `experiments/results/ablation_results.csv` "
                "does not exist yet, so there is nothing to tabulate.")
    lines = ["## Phase 5 — Ablations", "",
            "| Axis | Variant | mAP@0.5 | Δ vs. baseline | F1 |",
            "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['axis']} | {r['variant']} | {r['map50']} | "
                     f"{r.get('map50_pct_change_vs_baseline', '')}% | {r['f1']} |")
    return "\n".join(lines)


def plot_phase3a(phase3a_latest: dict):
    if not phase3a_latest:
        return
    models = list(phase3a_latest)
    labels = [MODEL_LABELS.get(m, m) for m in models]
    metrics = ["map50", "map50_95", "precision", "recall", "f1"]
    metric_labels = ["mAP@0.5", "mAP@0.5:.95", "Precision", "Recall", "F1"]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(metrics))
    width = 0.8 / max(len(models), 1)
    for i, m in enumerate(models):
        values = [float(phase3a_latest[m].get(k) or 0) for k in metrics]
        ax.bar([xi + i * width for xi in x], values, width=width, label=labels[i])
    ax.set_xticks([xi + width * (len(models) - 1) / 2 for xi in x])
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score")
    ax.set_title("Phase 3a — In-Domain Baseline Comparison (SOURCE test)")
    ax.legend()
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "phase3a_comparison.png", dpi=150)
    plt.close(fig)


def plot_cross_country(by_model_country: dict):
    if not by_model_country:
        return
    models = list(by_model_country)
    countries_present = [c for c in COUNTRY_ORDER
                         if any(c in by_model_country[m] for m in models)]
    if not countries_present:
        return

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = range(len(countries_present))
    width = 0.8 / max(len(models), 1)
    for i, m in enumerate(models):
        values = []
        for c in countries_present:
            r = by_model_country[m].get(c)
            values.append(float(r["map50"]) if r and r.get("map50") not in ("", None) else 0)
        bars = ax.bar([xi + i * width for xi in x], values, width=width,
                      label=MODEL_LABELS.get(m, m))
        # Mark bars for countries this model has no data for, so a 0-height
        # bar never reads as "measured zero" -- it means PENDING or N/A.
        for xi, c, v in zip(x, countries_present, values):
            if c not in by_model_country[m]:
                ax.text(xi + i * width, 0.01, "N/A", ha="center", va="bottom",
                        fontsize=8, rotation=90, color="gray")

    ax.set_xticks([xi + width * (len(models) - 1) / 2 for xi in x])
    ax.set_xticklabels([COUNTRY_LABELS.get(c, c) for c in countries_present],
                       rotation=20, ha="right")
    ax.set_ylabel("mAP@0.5")
    ax.set_title("Phase 4 — Cross-Country Zero-Shot Generalization")
    ax.legend()
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "cross_country_map50.png", dpi=150)
    plt.close(fig)


def main():
    log_rows = read_csv_rows(LOG_PATH)
    cc_rows = read_csv_rows(CROSS_COUNTRY_PATH)

    phase3a_md, phase3a_latest = build_phase3a_table(log_rows)
    cross_country_md, by_model_country = build_cross_country_table(cc_rows)
    ablation_md = build_ablation_table()

    plot_phase3a(phase3a_latest)
    plot_cross_country(by_model_country)

    # Loud, top-of-document banner whenever any PRELIMINARY (time-boxed
    # subsample) caveat applies to a country actually present in this run --
    # the per-row footnote is easy to skim past; this isn't.
    countries_present = {r["dataset"] for r in cc_rows}
    prelim_countries = [c for c in countries_present
                        if c in CAVEATS and "PRELIMINARY" in CAVEATS[c]]
    banner = ""
    if prelim_countries:
        names = ", ".join(COUNTRY_LABELS.get(c, c) for c in sorted(prelim_countries))
        banner = (
            f"\n> ⚠️ **PRELIMINARY RESULTS for {names}.** Faster R-CNN was evaluated "
            f"on a small time-boxed random subsample for these countries, not the "
            f"full target set. Treat these numbers as directional only -- re-run "
            f"at full scale (`src/eval/cross_country_eval.py`'s `DATASET_SAMPLE_SIZES`) "
            f"before using them in the report. See the per-country caveats below "
            f"for exact sample sizes.\n"
        )

    out_path = REPORTS_DIR / "results_tables.md"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    content = f"""# Results Tables — Auto-Generated

Generated by `src/eval/generate_report_tables.py` from `experiments/results/`.
Rerun this script any time new results land (e.g. after merging a Colab run's
output) -- it always reflects whatever is currently on disk, and marks
anything missing as PENDING rather than omitting it silently.

**Do not hand-edit this file** -- rerun the script instead, so it never goes
stale relative to `experiment_log.csv`.
{banner}
{phase3a_md}

{cross_country_md}

{ablation_md}

## Figures

- `reports/figures/phase3a_comparison.png`
- `reports/figures/cross_country_map50.png`
"""
    out_path.write_text(content, encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Wrote {FIGURES_DIR / 'phase3a_comparison.png'}" if phase3a_latest else "  (no Phase 3a figure -- no data yet)")
    print(f"Wrote {FIGURES_DIR / 'cross_country_map50.png'}" if by_model_country else "  (no cross-country figure -- no data yet)")


if __name__ == "__main__":
    main()
