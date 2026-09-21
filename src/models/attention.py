"""
Coordinate Attention (Hou et al., CVPR 2021) -- PHASE 3b.

WHY COORDINATE ATTENTION AND NOT CBAM (this is the viva answer -- do not
substitute CBAM): three of the four classes (D00 longitudinal, D10 transverse,
D20 alligator) are directional, elongated structures. Coordinate attention
factorizes global pooling into two 1D poolings, one along each spatial axis
(H and W), so it keeps axis-wise positional information instead of collapsing
it. CBAM's spatial-attention branch instead pools across channels into a
single 2D map, discarding exactly that directional structure.

ALGORITHM (matches the paper; nn.SiLU is used for the shared-MLP activation
instead of the paper's h-swish, to stay consistent with the rest of YOLOv8,
which uses SiLU everywhere -- Conv.default_act = nn.SiLU()):
  1. Pool the input separately along H and along W: (B,C,H,W) -> (B,C,H,1) and
     (B,C,1,W). Two direction-aware descriptors, not one global vector.
  2. Concatenate along the spatial axis, pass through a shared 1x1 conv + BN +
     SiLU to a reduced channel count (channels // reduction).
  3. Split back into the H and W branches; project each to the ORIGINAL
     channel count with its own 1x1 conv; sigmoid.
  4. Multiply the input by both attention maps (broadcast over the axis each
     one doesn't have positional resolution for).
Output shape == input shape: this is a channel-preserving recalibration, not
a downsampling or channel-changing layer, which is what makes it safe to drop
into the architecture at a single point without touching anything downstream.

REGISTRATION (CLAUDE.md section 3: "Register the CA module in the Ultralytics
model parser and reference it from a custom model YAML. Import alone will not
work and fails silently."):
ultralytics.nn.tasks.parse_model() resolves a YAML module name string via
`globals()[m]` -- literally a lookup in tasks.py's OWN module namespace, not
in whatever file imported this one. So `from attention import
CoordinateAttention` inside another script does nothing for that lookup; the
name has to be injected into tasks.py's globals directly. register_ca_module()
below does that. It must run before YOLO(<custom_yaml>) is constructed.
Verified against the installed ultralytics 8.4.x source
(ultralytics/nn/tasks.py, parse_model(), the `else: c2 = ch[f]` branch --
see yolo_attention.py for why that matters for channel-count arguments).
"""

import torch
import torch.nn as nn


class CoordinateAttention(nn.Module):
    """Channel-preserving attention block: recalibrates a (B,C,H,W) feature
    map using two direction-aware pooled descriptors instead of one global one.

    `channels` is NOT inferred at construction time -- ultralytics' model
    parser does not auto-compute a channel arg for modules outside its
    `base_modules` set (this one included), so the custom model YAML must pass
    the correct already-width-scaled channel count explicitly. For YOLOv8n
    (width_multiple=0.25) the value at the intended insertion point (end of
    backbone, i.e. right after SPPF) is 256 -- see config/yolov8n_ca.yaml.
    The assertion in forward() catches a wrong value immediately (on the
    first forward pass) rather than letting it silently produce garbage.
    """

    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()
        self.channels = channels

        # Pool each axis to size 1 while keeping the other axis's resolution,
        # so H-position and W-position information survive separately.
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # (B,C,H,W) -> (B,C,H,1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # (B,C,H,W) -> (B,C,1,W)

        mip = max(8, channels // reduction)  # bottleneck width, floor of 8
        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()

        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.channels, (
            f"CoordinateAttention configured for {self.channels} channels but "
            f"received {x.shape[1]}. This module is scale-specific -- it was "
            f"sized for one particular YOLOv8 scale (see config/yolov8n_ca.yaml). "
            f"Using this yaml with a different scale (s/m/l/x) or backbone width "
            f"will hit this assertion instead of silently training on garbage."
        )
        identity = x
        _, _, h, w = x.shape

        x_h = self.pool_h(x)                        # (B, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)     # (B, C, W, 1) -- align spatial dim with x_h's

        y = torch.cat([x_h, x_w], dim=2)             # (B, C, H+W, 1)
        y = self.act(self.bn1(self.conv1(y)))        # (B, mip, H+W, 1)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)                # back to (B, mip, 1, W)

        a_h = torch.sigmoid(self.conv_h(x_h))        # (B, C, H, 1) -- per-row gate
        a_w = torch.sigmoid(self.conv_w(x_w))        # (B, C, 1, W) -- per-column gate

        # Broadcast multiply: every pixel is scaled by (its row's gate) x
        # (its column's gate), so the recalibration is position-aware along
        # both axes independently -- the property CBAM's pooled 2D map lacks.
        return identity * a_h * a_w


def register_ca_module():
    """Make CoordinateAttention resolvable by name inside YOLOv8 model YAMLs.

    Must be called before any `YOLO(<yaml with CoordinateAttention in it>)`
    construction. Idempotent -- safe to call more than once.
    """
    import ultralytics.nn.tasks as tasks

    tasks.CoordinateAttention = CoordinateAttention
