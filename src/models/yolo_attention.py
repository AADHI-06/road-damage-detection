"""
Attention-integrated YOLOv8 variant -- PHASE 3b (the project spec section 3).

build_ca_model() is the single entry point train_yolo.py uses (via
--attention) to construct the CA-augmented network instead of plain
yolov8n.pt. Reuses every other piece of train_yolo.py unchanged (chunked
training, resume, eval, logging), so 3b differs from 3a in exactly one
variable -- the architecture -- which is the whole point of the ablation.

VALIDATION BEFORE TRUSTING ANY 3b RESULT (this file's own stub used to warn
about exactly this): a silently-ignored custom module still builds and trains
-- ultralytics doesn't error on an unused registration, it just never uses it
-- so "the run completed" is not evidence CA was actually there.
validate_ca_present() checks for a real CoordinateAttention instance in the
built model and a parameter-count delta against plain yolov8n, and is called
automatically every time build_ca_model() runs.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from attention import CoordinateAttention, register_ca_module  # noqa: E402

CA_YAML = REPO_ROOT / "config" / "yolov8n_ca.yaml"


def count_ca_layers(model) -> int:
    """Number of CoordinateAttention instances actually present in a built model."""
    return sum(1 for m in model.modules() if isinstance(m, CoordinateAttention))


def validate_ca_present(model, plain_param_count: int | None = None):
    """Raise if CoordinateAttention did not make it into the built model.

    Checks TWO independent things, because either alone can be misleading:
      1. At least one CoordinateAttention module instance exists in the graph
         (catches "the yaml line was silently unresolved").
      2. Exactly one instance exists (the project spec section 3: "Insert CA at ONE
         location only... multiple insertions make the ablation
         uninterpretable" -- catches a copy-paste duplicate insertion, not
         just a missing one).
      3. If a reference plain-model parameter count is supplied, the CA
         model must have strictly more parameters (catches a CA layer that
         built but is a no-op / has 0 trainable params due to a wiring bug).
    """
    n = count_ca_layers(model)
    if n == 0:
        raise RuntimeError(
            "CoordinateAttention was NOT found anywhere in the built model. "
            "The yaml line resolved to nothing, or register_ca_module() was "
            "not called before construction -- see attention.py's docstring "
            "on why a bare import does not register the module."
        )
    if n > 1:
        raise RuntimeError(
            f"Found {n} CoordinateAttention instances, expected exactly 1 "
            f"(the project spec section 3: single insertion point only)."
        )

    if plain_param_count is not None:
        ca_param_count = sum(p.numel() for p in model.parameters())
        if ca_param_count <= plain_param_count:
            raise RuntimeError(
                f"CA model has {ca_param_count:,} params, plain model has "
                f"{plain_param_count:,} -- CA should add parameters, not zero "
                f"or fewer. The layer built but may not be contributing anything."
            )
        added = ca_param_count - plain_param_count
        print(f"  CA validation OK: {n} CoordinateAttention layer, "
              f"+{added:,} params over plain yolov8n ({plain_param_count:,} -> {ca_param_count:,})")
    else:
        print(f"  CA validation OK: {n} CoordinateAttention layer present")


def build_ca_model(pretrained_weights: str | None = "yolov8n.pt", validate: bool = True):
    """Construct YOLOv8n-with-CoordinateAttention, optionally warm-started
    from COCO-pretrained yolov8n weights (matching-shape layers only --
    ultralytics' .load() skips whatever doesn't match, which is exactly the
    new CA layer plus the tail Detect layer if nc differs).

    Returns an ultralytics YOLO instance, ready for .train(...).
    """
    register_ca_module()  # must happen before YOLO(<yaml>) parses the model

    from ultralytics import YOLO

    model = YOLO(str(CA_YAML))

    plain_param_count = None
    if validate:
        try:
            # DetectionModel (not the high-level YOLO() wrapper) so nc can be
            # forced to match the CA model's -- comparing against stock
            # yolov8n.yaml's default nc=80 would be misleading: an 80-class
            # detection head has far more parameters than a 4-class one, and
            # that difference dwarfs CA's contribution, hiding the very
            # thing this check exists to catch.
            from ultralytics.nn.tasks import DetectionModel

            nc = model.model.yaml.get("nc", 4)
            plain = DetectionModel("yolov8n.yaml", nc=nc, verbose=False)
            plain_param_count = sum(p.numel() for p in plain.parameters())
        except Exception as e:  # pragma: no cover -- validation nicety, not required for training
            print(f"  (skipping plain-model param comparison: {e})")
        validate_ca_present(model.model, plain_param_count)

    if pretrained_weights:
        model.load(pretrained_weights)

    return model
