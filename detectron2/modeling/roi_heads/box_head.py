# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
from typing import List
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm, Linear
from detectron2.utils.registry import Registry

__all__ = ["CAMBoxHeadConv", "FastRCNNConvFCHead", "build_box_head", "ROI_BOX_HEAD_REGISTRY"]

ROI_BOX_HEAD_REGISTRY = Registry("ROI_BOX_HEAD")
ROI_BOX_HEAD_REGISTRY.__doc__ = """
Registry for box heads, which make box predictions from per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""


@ROI_BOX_HEAD_REGISTRY.register()
class CAMBoxHeadConv(nn.Module):
    """
    Copyright (c) Facebook, Inc. and its affiliates.
    Adapted Detectron2 class.

    A head with several 3x3 conv layers (each followed by norm & relu) and
    several fc layers (each followed by relu) that allows calculating class activation maps (CAMs).
    """

    def __init__(self, cfg, input_shape, num_classes, cls_agnostic_bbox_reg, box_dim=4):
        """
        The following attributes are parsed from config:
            num_conv, num_fc: the number of conv/fc layers
            conv_dim/fc_dim: the dimension of the conv/fc layers
            norm: normalization for the conv layers
        """
        super().__init__()

        # fmt: off
        num_conv = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV # 4
        conv_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM # 256
        num_fc = cfg.MODEL.ROI_BOX_HEAD.NUM_FC # 0
        fc_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM # 1024
        norm = cfg.MODEL.ROI_BOX_HEAD.NORM # 
        # fmt: on
        assert num_conv + num_fc > 0

        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        self.num_classes = num_classes
        self.pred_reg = num_bbox_reg_classes * box_dim

        # 256, 7, 7
        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)

        # 256-512-1024-1024-1024
        # final size (b, 1024, 7, 7)
        self.conv_norm_relus = []
        for k in range(num_conv):
            dim_in = conv_dim
            dim_out = conv_dim
            conv = Conv2d(
                dim_in,
                dim_out,
                kernel_size=3,
                padding=1,
                bias=not norm,
                norm=get_norm(norm, dim_out),
                activation=F.relu,
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)

        # 256, 7, 7
        self._output_size = (conv_dim, self._output_size[1], self._output_size[2])

        # NOT USE
        self.fcs = []
        for k in range(num_fc):  # 0
            fc = Linear(np.prod(self._output_size), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim

        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        for i, layer in enumerate(self.conv_norm_relus):
            if i==0:
                features = layer(x)
                x = features
            else:
                x= layer(x)

        return x, features

    @property
    @torch.jit.unused
    def output_shape(self):
        """
        Returns:
            ShapeSpec: the output feature shape
        """
        o = self._output_size
        if isinstance(o, int):
            return ShapeSpec(channels=o)
        else:
            return ShapeSpec(channels=o[0], height=o[1], width=o[2]) # 256, 7, 7


# To get torchscript support, we make the head a subclass of `nn.Sequential`.
# Therefore, to add new layers in this head class, please make sure they are
# added in the order they will be used in forward().
@ROI_BOX_HEAD_REGISTRY.register()
class FastRCNNConvFCHead(nn.Sequential):
    """
    A head with several 3x3 conv layers (each followed by norm & relu) and then
    several fc layers (each followed by relu).
    """

    @configurable
    def __init__(
        self, input_shape: ShapeSpec, *, conv_dims: List[int], fc_dims: List[int], conv_norm=""
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature.
            conv_dims (list[int]): the output dimensions of the conv layers
            fc_dims (list[int]): the output dimensions of the fc layers
            conv_norm (str or callable): normalization for the conv layers.
                See :func:`detectron2.layers.get_norm` for supported types.
        """
        super().__init__()
        assert len(conv_dims) + len(fc_dims) > 0

        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)

        self.conv_norm_relus = []
        for k, conv_dim in enumerate(conv_dims):
            conv = Conv2d(
                self._output_size[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=nn.ReLU(),
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])

        self.fcs = []
        for k, fc_dim in enumerate(fc_dims):
            if k == 0:
                self.add_module("flatten", nn.Flatten())
            fc = nn.Linear(int(np.prod(self._output_size)), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.add_module("fc_relu{}".format(k + 1), nn.ReLU())
            self.fcs.append(fc)
            self._output_size = fc_dim

        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    @classmethod
    def from_config(cls, cfg, input_shape):
        num_conv = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
        conv_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
        num_fc = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
        fc_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        return {
            "input_shape": input_shape,
            "conv_dims": [conv_dim] * num_conv,
            "fc_dims": [fc_dim] * num_fc,
            "conv_norm": cfg.MODEL.ROI_BOX_HEAD.NORM,
        }

    def forward(self, x):
        #import pdb;pdb.set_trace()
        #from fvcore.nn import FlopCountAnalysis
        total_flops = 0
        for layer in self:
            #total_flops += FlopCountAnalysis(layer, x).total()
            x = layer(x)

        #print(total_flops)
        #import pdb;pdb.set_trace()
        return x

    @property
    @torch.jit.unused
    def output_shape(self):
        """
        Returns:
            ShapeSpec: the output feature shape
        """
        o = self._output_size
        if isinstance(o, int):
            return ShapeSpec(channels=o)
        else:
            return ShapeSpec(channels=o[0], height=o[1], width=o[2])


# def build_box_head(cfg, input_shape):
#     """
#     Build a box head defined by `cfg.MODEL.ROI_BOX_HEAD.NAME`.
#     """
#     name = cfg.MODEL.ROI_BOX_HEAD.NAME
#     num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
#     cls_agnostic = cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
#     return ROI_BOX_HEAD_REGISTRY.get(name)(cfg, input_shape, num_classes, cls_agnostic)

# original
def build_box_head(cfg, input_shape):
    """
    Build a box head defined by `cfg.MODEL.ROI_BOX_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_BOX_HEAD.NAME
    return ROI_BOX_HEAD_REGISTRY.get(name)(cfg, input_shape)