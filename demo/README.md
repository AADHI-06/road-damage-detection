# Road Damage Detector — Demo (Phase 7)

**Status: built and verified locally.** Simple single-page UI: one card,
drag-and-drop, no marketing sections.

## What it does

Upload one road image → the YOLO26n checkpoint
(`experiments/runs/yolo26n_source/weights/best.pt`) runs inference → boxes
are drawn on the image, labeled with damage type (D00/D10/D20/D40) and
confidence, plus a small table listing every detection.

## Run it

```bash
pip install -r demo/requirements.txt
python demo/app.py
```

Open **http://127.0.0.1:7860**.

## Verified

Re-verified after switching the backing checkpoint from YOLOv8n to YOLO26n
(direct model load + inference, same test image the original YOLOv8n
verification used): on `data/India/train/images/India_001744.jpg` (a known
4-pothole image from the Phase 2 sanity check), YOLO26n correctly detects
2 of the 4 potholes (confidences 0.561, 0.543) — fewer than YOLOv8n's
earlier 3/4 on the same image, consistent with YOLO26n's slightly lower
in-domain recall (Chapter 7 of the project report). The full
upload → `/api/predict` → render flow was not re-tested through the browser
after this swap; only the model-load-and-inference path was re-confirmed.

## Files

```
demo/
├── app.py              Flask backend: loads the model once, serves the
│                        page, exposes POST /api/predict
├── static/
│   ├── index.html       the page
│   ├── style.css
│   └── script.js         upload, drag-drop, calls /api/predict, renders result
└── requirements.txt      flask, ultralytics, pillow
```

## Hosted URL

`NOT YET DEPLOYED`
