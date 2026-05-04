# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn.modules.utils import _pair

# Always import TorchVision op as a safe fallback.
from torchvision.ops import roi_pool as _tv_roi_pool

_HAS_C = False
try:
    from fasterRCNN.lib.model import _C  # type: ignore
    from torch.autograd import Function
    from torch.autograd.function import once_differentiable

    _HAS_C = True

    class _ROIPool(Function):
        @staticmethod
        def forward(ctx, input, roi, output_size, spatial_scale):
            ctx.output_size = _pair(output_size)
            ctx.spatial_scale = spatial_scale
            ctx.input_shape = input.size()
            output, argmax = _C.roi_pool_forward(
                input, roi, spatial_scale, output_size[0], output_size[1]
            )
            ctx.save_for_backward(input, roi, argmax)
            return output

        @staticmethod
        @once_differentiable
        def backward(ctx, grad_output):
            input, rois, argmax = ctx.saved_tensors
            output_size = ctx.output_size
            spatial_scale = ctx.spatial_scale
            bs, ch, h, w = ctx.input_shape
            grad_input = _C.roi_pool_backward(
                grad_output,
                input,
                rois,
                argmax,
                spatial_scale,
                output_size[0],
                output_size[1],
                bs,
                ch,
                h,
                w,
            )
            return grad_input, None, None, None

    _roi_pool_c = _ROIPool.apply
except Exception:
    _HAS_C = False
    _roi_pool_c = None  # type: ignore


def roi_pool(input, roi, output_size, spatial_scale):
    """
    Robust ROIPool:
    - If the vendored C++/CUDA op exists *and* tensors are CUDA, use it.
    - Otherwise fall back to TorchVision's roi_pool (works on CPU and CUDA).
    """
    if _HAS_C and input.is_cuda and roi.is_cuda:
        return _roi_pool_c(input, roi, output_size, spatial_scale)  # type: ignore[misc]
    return _tv_roi_pool(
        input,
        roi,
        output_size=_pair(output_size),
        spatial_scale=spatial_scale,
    )


class ROIPool(nn.Module):
    def __init__(self, output_size, spatial_scale):
        super(ROIPool, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(self, input, rois):
        return roi_pool(input, rois, self.output_size, self.spatial_scale)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ")"
        return tmpstr
