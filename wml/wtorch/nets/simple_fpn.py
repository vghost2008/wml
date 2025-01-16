# Copyright (c) Facebook, Inc. and its affiliates.
import math
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F
from torch import nn
from wml.wtorch.nn import Conv2d
from wml.wtorch.nn import get_norm


class SimpleFPN(torch.nn.Module):
    """
    This module implements :paper:`FPN`.
    It creates pyramid features built on top of some input feature maps.
    """

    _fuse_type: torch.jit.Final[str]

    def __init__( self, in_channels, out_channels, norm="",  fuse_type="sum",interpolate_mode="nearest"
    ):
        """
        Args:
            out_channels (int): number of channels in the output feature maps.
            norm (str): the normalization to use.
            fuse_type (str): types for fusing the top down features and the lateral
                ones. It can be "sum" (default), which sums up element-wise; or "avg",
                which takes the element-wise mean of the two.
        """
        super().__init__()

        # Feature map strides and channels from the bottom up network (e.g. ResNet)
        in_channels_per_feature = in_channels
        self.interpolate_mode = interpolate_mode

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(in_channels_per_feature):
            lateral_norm = get_norm(norm, out_channels)
            output_norm = get_norm(norm, out_channels)

            lateral_conv = Conv2d(
                in_channels, out_channels, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            output_conv = Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
            )
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            self.add_module("fpn_lateral{}".format(idx), lateral_conv)
            self.add_module("fpn_output{}".format(idx), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]

        assert fuse_type in {"avg", "sum"}
        self._fuse_type = fuse_type

    def forward(self, xs):
        """
        Args:
            xs: list of tensors, shape like [H,W],[H//2,W//2],[H//4,W//4],...

        Returns:
            list of tensors
        """
        results = []
        prev_features = self.lateral_convs[0](xs[-1])
        results.append(self.output_convs[0](prev_features))

        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, (lateral_conv, output_conv) in enumerate(
            zip(self.lateral_convs, self.output_convs)
        ):
            # Slicing of ModuleList is not supported https://github.com/pytorch/pytorch/issues/47336
            # Therefore we loop over all modules but skip the first one
            if idx > 0:
                features = xs[-idx - 1]
                if self.interpolate_mode == "nearest":
                    top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode="nearest")
                else:
                    top_down_features = F.interpolate(prev_features, size=features.shape[-2:],mode="bilinear")
                lateral_features = lateral_conv(features)
                prev_features = lateral_features + top_down_features
                if self._fuse_type == "avg":
                    prev_features /= 2
                results.insert(0, output_conv(prev_features))

        return results
