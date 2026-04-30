# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
This repo vendors parts of Detectron2-style ROI ops that normally rely on a
compiled C++ extension (`fasterRCNN.lib.model._C`).

On macOS (and in many non-CUDA environments) that extension isn't available.
We fall back to TorchVision's pure-extension wheels.
"""

try:
    from fasterRCNN.lib.model import _C  # type: ignore

    nms = _C.nms
except Exception:
    import torch
    from torchvision.ops import nms as _tv_nms

    def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float):
        return _tv_nms(boxes, scores, iou_threshold)
