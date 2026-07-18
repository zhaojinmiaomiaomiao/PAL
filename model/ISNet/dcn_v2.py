#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import math

import torch
from torch import nn
from torch.nn.modules.utils import _pair
from torchvision.ops import deform_conv2d


class DCNv2(nn.Module):
    """
    torchvision-based replacement for the original _ext-backed DCNv2.
    Keeps the same forward signature: forward(input, offset, mask)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation=1,
        deformable_groups=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()

    def forward(self, input, offset, mask):
        kh, kw = self.kernel_size

        assert 2 * self.deformable_groups * kh * kw == offset.shape[1], \
            f"offset channels mismatch: got {offset.shape[1]}"
        assert self.deformable_groups * kh * kw == mask.shape[1], \
            f"mask channels mismatch: got {mask.shape[1]}"

        return deform_conv2d(
            input=input,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )


class DCN(DCNv2):
    """
    Standard DCN wrapper:
    - predicts offset and mask by a conv layer
    - then applies deform_conv2d
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation=1,
        deformable_groups=1,
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, deformable_groups
        )

        channels_ = self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            channels_,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input):
        out = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        return deform_conv2d(
            input=input,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )


class DCNv2Pooling(nn.Module):
    """
    Placeholder only.

    The original implementation depends on custom compiled ops:
      - dcn_v2_psroi_pooling_forward
      - dcn_v2_psroi_pooling_backward

    torchvision has no direct drop-in replacement for this exact op.
    """

    def __init__(
        self,
        spatial_scale,
        pooled_size,
        output_dim,
        no_trans,
        group_size=1,
        part_size=None,
        sample_per_part=4,
        trans_std=0.0,
    ):
        super().__init__()
        self.spatial_scale = spatial_scale
        self.pooled_size = pooled_size
        self.output_dim = output_dim
        self.no_trans = no_trans
        self.group_size = group_size
        self.part_size = pooled_size if part_size is None else part_size
        self.sample_per_part = sample_per_part
        self.trans_std = trans_std

    def forward(self, input, rois, offset):
        raise NotImplementedError(
            "DCNv2Pooling depends on the original compiled _ext backend and "
            "has no direct torchvision drop-in replacement. "
            "If your project uses this class, you need to either "
            "(1) compile the original extension, or "
            "(2) rewrite this part with roi_align / another approximation."
        )


class DCNPooling(DCNv2Pooling):
    """
    Placeholder only for the original deformable PSRoI pooling module.
    """

    def __init__(
        self,
        spatial_scale,
        pooled_size,
        output_dim,
        no_trans,
        group_size=1,
        part_size=None,
        sample_per_part=4,
        trans_std=0.0,
        deform_fc_dim=1024,
    ):
        super().__init__(
            spatial_scale,
            pooled_size,
            output_dim,
            no_trans,
            group_size,
            part_size,
            sample_per_part,
            trans_std,
        )
        self.deform_fc_dim = deform_fc_dim

        if not no_trans:
            self.offset_mask_fc = nn.Sequential(
                nn.Linear(
                    self.pooled_size * self.pooled_size * self.output_dim,
                    self.deform_fc_dim,
                ),
                nn.ReLU(inplace=True),
                nn.Linear(self.deform_fc_dim, self.deform_fc_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.deform_fc_dim, self.pooled_size * self.pooled_size * 3),
            )
            self.offset_mask_fc[4].weight.data.zero_()
            self.offset_mask_fc[4].bias.data.zero_()

    def forward(self, input, rois):
        raise NotImplementedError(
            "DCNPooling depends on the original compiled _ext backend and "
            "has no direct torchvision drop-in replacement. "
            "If your project uses this class, you need to either "
            "(1) compile the original extension, or "
            "(2) rewrite this part with roi_align / another approximation."
        )