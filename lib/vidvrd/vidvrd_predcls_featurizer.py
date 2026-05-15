"""
VIDVRD predcls-style featurizer.

This module converts:
  - frames tensor im_data [T,3,H,W]
  - boxes [N,5] with frame indices
  - pair_idx [R,2] and im_idx [R]
into the tensors needed by `lib/sttran.py`:
  - features [N,2048]
  - union_feat [R,1024,7,7]
  - spatial_masks [R,2,27,27]

It reuses the Faster R-CNN components already loaded in `lib/object_detector.detector`.

Implementation is intentionally close to the predcls branch in `lib/object_detector.py`
so that dimensions match the pretrained checkpoint expectations.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from lib.object_detector import draw_union_boxes  # same rasterizer used elsewhere
from lib.vidvrd.vidvrd_featurizer_types import FeaturizerOutput


class VidvrdPredclsFeaturizer(nn.Module):
    """
    Wraps a Faster R-CNN model (resnet-101 in this repo) to compute ROI features
    for GT boxes and GT pairs.

    Notes:
      - This featurizer does NOT build boxes/pairs. It only featurizes them.
      - Boxes are expected in original pixel coordinates; this module applies `scale`
        (im_info[0,2]) in the same way as `lib/object_detector.py`.
    """

    def __init__(self, fasterRCNN: nn.Module, chunk_frames: int = 10):
        super().__init__()
        self.fasterRCNN = fasterRCNN
        self.chunk_frames = int(chunk_frames)

    @torch.no_grad()
    def forward(
        self,
        *,
        im_data: torch.Tensor,   # [T,3,H,W]
        im_info: torch.Tensor,   # [T,3] or [1,3]; scale is im_info[0,2]
        boxes: torch.Tensor,     # [N,5]  [t,x1,y1,x2,y2]
        pair_idx: torch.Tensor,  # [R,2]
        im_idx: torch.Tensor,    # [R] frame index per pair (float or long)
        spatial_size: int = 27,
    ) -> FeaturizerOutput:
        device = im_data.device
        T = int(im_data.shape[0])

        # Ensure tensors are on the same device as the backbone feature maps.
        # In notebook code it is common to create boxes/pairs on CPU (e.g. from numpy),
        # which makes the CUDA ROI ops fail with "rois must be a CUDA tensor".
        if im_info.device != device:
            im_info = im_info.to(device)
        if boxes.device != device:
            boxes = boxes.to(device)
        if pair_idx.device != device:
            pair_idx = pair_idx.to(device)
        if im_idx.device != device:
            im_idx = im_idx.to(device)

        # Normalize im_idx dtype for indexing / concatenation.
        if im_idx.dtype.is_floating_point:
            im_idx_long = im_idx.long()
        else:
            im_idx_long = im_idx

        # Backbone feature maps per frame: [T,1024,H',W']
        base_feats = []
        counter = 0
        while counter < T:
            end = min(T, counter + self.chunk_frames)
            base_feat = self.fasterRCNN.RCNN_base(im_data[counter:end])
            base_feats.append(base_feat)
            counter = end
        fmaps = torch.cat(base_feats, dim=0)

        # Scale boxes for ROIAlign, following object_detector.py logic.
        scale = float(im_info.reshape(-1, im_info.shape[-1])[0, 2].item())
        boxes_scaled = boxes.clone()
        boxes_scaled[:, 1:] = boxes_scaled[:, 1:] * scale

        # Node ROI features: ROIAlign -> head_to_tail -> [N,2048]
        node_roi = self.fasterRCNN.RCNN_roi_align(fmaps, boxes_scaled)
        features = self.fasterRCNN._head_to_tail(node_roi)

        # Union boxes for each pair: [R,5] = [im_idx, x1,y1,x2,y2]
        s = pair_idx[:, 0].long()
        o = pair_idx[:, 1].long()
        union_boxes = torch.cat(
            (
                im_idx_long[:, None].to(dtype=boxes_scaled.dtype),
                torch.min(boxes_scaled[:, 1:3][s], boxes_scaled[:, 1:3][o]),
                torch.max(boxes_scaled[:, 3:5][s], boxes_scaled[:, 3:5][o]),
            ),
            dim=1,
        )

        union_feat = self.fasterRCNN.RCNN_roi_align(fmaps, union_boxes)

        # Spatial masks are computed in unscaled coordinate space in this repo:
        # predcls code divides FINAL_BBOXES back by scale before rasterization.
        boxes_unscaled = boxes_scaled.clone()
        if scale != 0:
            boxes_unscaled[:, 1:] = boxes_unscaled[:, 1:] / scale

        pair_rois = torch.cat((boxes_unscaled[s, 1:], boxes_unscaled[o, 1:]), dim=1).detach().cpu().numpy()
        spatial_masks = torch.tensor(draw_union_boxes(pair_rois, spatial_size) - 0.5, device=device, dtype=torch.float32)

        return FeaturizerOutput(features=features, union_feat=union_feat, spatial_masks=spatial_masks)

