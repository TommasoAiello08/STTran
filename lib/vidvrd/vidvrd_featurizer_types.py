"""Shared ROI featurizer output container (no Faster R-CNN imports)."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FeaturizerOutput:
    features: torch.Tensor       # [N,2048]
    union_feat: torch.Tensor     # [R,1024,7,7]
    spatial_masks: torch.Tensor  # [R,2,27,27]
