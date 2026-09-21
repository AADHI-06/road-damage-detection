"""
Single place where the Faster R-CNN architecture is constructed.

Training and evaluation MUST build the detector identically. When each had its
own copy of this code, any later edit to one (image size, box head, anchor
settings) would silently produce a model at eval time that differs from the one
that was trained -- and the only symptom would be quietly wrong metrics. One
function, used by both, removes that failure mode.
"""

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_faster_rcnn(num_classes: int, imgsz: int, pretrained: bool = False,
                      checkpoint=None):
    """Faster R-CNN ResNet-50 FPN with a `num_classes`-way box head.

    Args:
        num_classes: INCLUDING background. This project has 4 damage classes,
            so it passes 5 (torchvision reserves label 0 for background).
        imgsz: fixed min_size == max_size so every image is resized the same
            way, comparably to YOLO's fixed imgsz.
        pretrained: True for training (COCO-pretrained detector -- training a
            ResNet-50 detector from scratch on ~12.7k images would badly
            underfit, and every CRDDC-2022 entry used pretrained weights).
            False when a trained checkpoint is about to be loaded over it.
        checkpoint: optional path to a checkpoint saved by train_faster_rcnn.py.

    Note on weights_backbone: torchvision defaults it to the ImageNet ResNet-50
    weights and downloads ~98 MB even when weights=None. When a checkpoint is
    being loaded, every one of those tensors is immediately overwritten, so the
    download is pure waste and makes evaluation require internet access.
    """
    if pretrained:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights="DEFAULT", min_size=imgsz, max_size=imgsz,
        )
    else:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=None, weights_backbone=None, min_size=imgsz, max_size=imgsz,
        )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if checkpoint is not None:
        state = torch.load(str(checkpoint), map_location="cpu")
        model.load_state_dict(state["model"] if "model" in state else state)

    return model
