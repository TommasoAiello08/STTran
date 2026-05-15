"""
CPU- and CI-friendly stand-in for :class:`lib.vidvrd.vidvrd_predcls_featurizer.VidvrdPredclsFeaturizer`.

Use to verify JSON → ``build_vidvrd_predcls_entry`` → tensor shapes without
``faster_rcnn_ag.pth`` / CUDA ROI ops.
"""

from __future__ import annotations

import torch

from lib.vidvrd.vidvrd_featurizer_types import FeaturizerOutput


class VidvrdMockFeaturizer(torch.nn.Module):
    """Returns zero tensors with the shapes STTran predcls expects."""

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        *,
        im_data: torch.Tensor,
        im_info: torch.Tensor,
        boxes: torch.Tensor,
        pair_idx: torch.Tensor,
        im_idx: torch.Tensor,
        spatial_size: int = 27,
    ) -> FeaturizerOutput:
        del im_info, im_idx, spatial_size  # unused; kept for API parity
        device = im_data.device
        n = int(boxes.shape[0])
        r = int(pair_idx.shape[0])
        return FeaturizerOutput(
            features=torch.zeros((n, 2048), device=device, dtype=torch.float32),
            union_feat=torch.zeros((r, 1024, 7, 7), device=device, dtype=torch.float32),
            spatial_masks=torch.zeros((r, 2, 27, 27), device=device, dtype=torch.float32),
        )
