import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, get_norm
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from detectron2.layers import interpolate, get_instances_contour_interior
from pytorch_toolbelt import losses as L

from pytorch_toolbelt.modules import AddCoords
from detectron2.layers.dice_loss import DiceBCELoss, TverskyLoss
from detectron2.layers.transformer import *

import copy
from typing import Optional, List
from torch import Tensor
from detectron2.utils.misc import get_masks_from_tensor_list, nested_tensor_from_tensor_list, NestedTensor
from detectron2.layers.position_encoding import PositionEmbeddingSine, PositionEmbeddingLearned
from detectron2.layers.maskencoding import DctMaskEncoding
from detectron2.layers.mlp import MLP
from detectron2.modeling.roi_heads.aisformer.aisformer import AISFormer

ROI_MASK_HEAD_REGISTRY = Registry("ROI_MASK_HEAD")
ROI_MASK_HEAD_REGISTRY.__doc__ = """
Registry for mask heads, which predicts instance masks given
per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""

# """
import os
import matplotlib.pyplot as plt

def visualize_featmap(featmap, folder_name):
    path_name = os.path.join('../data/outtest/', folder_name)
    os.makedirs(path_name, exist_ok=True)
    np_featmap = featmap.detach().cpu().numpy()
    for i, fm in enumerate(np_featmap):
        plt.imsave(os.path.join(path_name, 'sample_{}.png'.format(str(i))),fm.mean(axis=0))

def visualize_prediction_logits(pred_logits, folder_name):
    path_name = os.path.join('../data/outtest/', folder_name)
    os.makedirs(path_name, exist_ok=True)
    np_pred_sm = pred_logits.sigmoid().detach().cpu().numpy()
    for i, logit_sm in enumerate(np_pred_sm):
        logit_mask = logit_sm > 0.5
        # print(os.path.join(path_name, 'sample_{}.png'.format(str(i))))
        plt.imsave(os.path.join(path_name, 'sample_{}.png'.format(str(i))), logit_mask)

def visualize_gt(gts, folder_name):
    path_name = os.path.join('../data/outtest/', folder_name)
    os.makedirs(path_name, exist_ok=True)

    np_gts = gts.detach().cpu().numpy()
    for i, gt in enumerate(np_gts):
        plt.imsave(os.path.join(path_name, 'sample_{}.png'.format(str(i))), gt)
# """

def mask_rcnn_loss(pred_mask_logits, instances, pred_mask_bo_logits,
                    use_i_mask=False, pred_a_mask_logits=None, use_justify_loss=False, c_iter=None, dice_loss=False,
                    pred_invisible_mask_logits=None):
    """
    """
    pass

def mask_rcnn_inference(pred_mask_logits, bo_mask_logits, bound_logits, bo_bound_logits, pred_instances):
    """
    Convert pred_mask_logits to estimated foreground probability masks while also
    extracting only the masks for the predicted classes in pred_instances. For each
    predicted box, the mask of the same class is attached to the instance by adding a
    new "pred_masks" field to pred_instances.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        pred_instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. Each Instances must have field "pred_classes".

    Returns:
        None. pred_instances will contain an extra "pred_masks" field storing a mask of size (Hmask,
            Wmask) for predicted class. Note that the masks are returned as a soft (non-quantized)
            masks the resolution predicted by the network; post-processing steps, such as resizing
            the predicted masks to the original image resolution and/or binarizing them, is left
            to the caller.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    #pred_mask_logits = pred_mask_logits[:,0:1]
    if cls_agnostic_mask:
        mask_probs_pred = pred_mask_logits.sigmoid()
        bound_probs_pred = bound_logits.sigmoid()
        bo_mask_probs_pred = bo_mask_logits.sigmoid()
        bo_bound_probs_pred = bo_bound_logits.sigmoid()
    else:
        # Select masks corresponding to the predicted classes
        num_masks = pred_mask_logits.shape[0]
        class_pred = cat([i.pred_classes for i in pred_instances])
        indices = torch.arange(num_masks, device=class_pred.device)
        mask_probs_pred = pred_mask_logits[indices, class_pred][:, None].sigmoid()
        bound_probs_pred = bound_logits.sigmoid()
        bo_mask_probs_pred = bo_mask_logits.sigmoid()
        bo_bound_probs_pred = bo_bound_logits.sigmoid()
    # mask_probs_pred.shape: (B, 1, Hmask, Wmask)

    num_boxes_per_image = [len(i) for i in pred_instances]
    mask_probs_pred = mask_probs_pred.split(num_boxes_per_image, dim=0)
    bo_mask_probs_pred = bo_mask_probs_pred.split(num_boxes_per_image, dim=0)
    bo_bound_probs_pred = bo_bound_probs_pred.split(num_boxes_per_image, dim=0)
    bound_probs_pred = bound_probs_pred.split(num_boxes_per_image, dim=0)

    for prob, instances in zip(mask_probs_pred, pred_instances):
        instances.pred_masks = prob  # (1, Hmask, Wmask)
        instances.raw_masks = prob

    for bo_prob, instances in zip(bo_mask_probs_pred, pred_instances):
        instances.pred_masks_bo = bo_prob  # (1, Hmask, Wmask)

    for bo_bound_prob, instances in zip(bo_bound_probs_pred, pred_instances):
        instances.pred_bounds_bo = bo_bound_prob  # (1, Hmask, Wmask)

    for bound_prob, instances in zip(bound_probs_pred, pred_instances):
        instances.pred_bounds = bound_prob  # (1, Hmask, Wmask)

def mask_rcnn_inference_amodal(pred_mask_logits, pred_a_mask_logits, pred_mask_logits_1, pred_a_mask_logits_1,
                                occluder_mask_logits, invisible_mask_logits, pred_instances):
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    #pred_mask_logits = pred_mask_logits[:,0:1]
    if cls_agnostic_mask:
        mask_probs_pred = pred_mask_logits.sigmoid()
        a_mask_probs_pred = pred_a_mask_logits.sigmoid()

        mask_probs_pred_1 = pred_mask_logits_1.sigmoid()
        a_mask_probs_pred_1 = pred_a_mask_logits_1.sigmoid()

        occluder_mask_probs_pred = occluder_mask_logits.sigmoid()
        invisible_mask_probs_pred = invisible_mask_logits.sigmoid()
    else:
        # Select masks corresponding to the predicted classes
        num_masks = pred_mask_logits.shape[0]
        class_pred = cat([i.pred_classes for i in pred_instances])
        indices = torch.arange(num_masks, device=class_pred.device)
        mask_probs_pred = pred_mask_logits[indices, class_pred][:, None].sigmoid()
        a_mask_probs_pred = pred_a_mask_logits[indices, class_pred][:, None].sigmoid()
        occluder_mask_probs_pred = occluder_mask_logits[indices, class_pred][:, None].sigmoid()
        invisible_mask_probs_pred = invisible_mask_logits[indices, class_pred][:, None].sigmoid()
    # mask_probs_pred.shape: (B, 1, Hmask, Wmask)

    num_boxes_per_image = [len(i) for i in pred_instances]
    mask_probs_pred = mask_probs_pred.split(num_boxes_per_image, dim=0)
    a_mask_probs_pred = a_mask_probs_pred.split(num_boxes_per_image, dim=0)

    mask_probs_pred_1 = mask_probs_pred_1.split(num_boxes_per_image, dim=0)
    a_mask_probs_pred_1 = a_mask_probs_pred_1.split(num_boxes_per_image, dim=0)

    occluder_mask_probs_pred = occluder_mask_probs_pred.split(num_boxes_per_image, dim=0)
    invisible_mask_probs_pred = invisible_mask_probs_pred.split(num_boxes_per_image, dim=0)

    for prob, prob_1, instances in zip(mask_probs_pred, mask_probs_pred_1, pred_instances):
        instances.pred_visible_masks = torch.relu(prob_1-prob) + prob

    for prob, prob_1, inv_prob, instances in zip(a_mask_probs_pred, a_mask_probs_pred_1, invisible_mask_probs_pred, pred_instances):
        a = torch.relu(prob_1-prob) + prob
        a_mask, _ = torch.max(cat([inv_prob, instances.pred_visible_masks], 1), dim =1 )
        a_mask = a_mask.unsqueeze(1)
        a_mask = torch.relu(a_mask-a) + a
        instances.pred_masks = a_mask
        instances.raw_masks = a_mask
        instances.pred_amodal_masks = a_mask

    for prob, instances in zip(occluder_mask_probs_pred, pred_instances):
        instances.pred_occluder_masks = prob  # (1, Hmask, Wmask)

    for prob, instances in zip(invisible_mask_probs_pred, pred_instances):
        instances.pred_invisible_masks = prob  # (1, Hmask, Wmask)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@ROI_MASK_HEAD_REGISTRY.register()
class MaskRCNNConvUpsampleHead(nn.Module):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_conv: the number of conv layers
            conv_dim: the dimension of the conv layers
            norm: normalization for the conv layers
        """
        super(MaskRCNNConvUpsampleHead, self).__init__()

        ###############
        if cfg.MODEL.ROI_MASK_HEAD.CUSTOM_NAME == 'AISFormer':
            self.mask_head_model = AISFormer(cfg, input_shape)
            print("AISFormer param: ", count_parameters(self.mask_head_model))
        else:
            assert False, "Invalid custom name for mask head"
        ###############

    def forward(self,x, c_iter, instances=None): # here
        #from fvcore.nn import FlopCountAnalysis
        #flops = FlopCountAnalysis(self.mask_head_model, x)
        #print(flops.total())
        return self.mask_head_model(x)

@ROI_MASK_HEAD_REGISTRY.register()
class BCNetConvGCNHead(nn.Module):
    """
    A mask head used in BCNet
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_conv: the number of conv layers
            conv_dim: the dimension of the conv layers
            norm: normalization for the conv layers
        """
        super(BCNetConvGCNHead, self).__init__()

        # fmt: off
        num_classes       = cfg.MODEL.ROI_HEADS.NUM_CLASSES # 7
        conv_dims         = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM # 256
        self.norm         = cfg.MODEL.ROI_MASK_HEAD.NORM # none or BN
        num_conv          = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV # 4
        input_channels    = input_shape.channels # 256
        cls_agnostic_mask = cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK
        # fmt: on

        self.visible_conv_norm_relus = []
        for k in range(num_conv):
            conv = Conv2d(
                512 if k == 0 else conv_dims,
                conv_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not self.norm,
                norm=get_norm(self.norm, conv_dims),
                activation=F.relu,
            )
            self.add_module("visible_fcn{}".format(k + 1), conv)
            self.visible_conv_norm_relus.append(conv)

        self.amodal_conv_norm_relus = []
        for k in range(num_conv+2):
            conv = Conv2d(
                1024 if k == 0 else conv_dims,
                conv_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not self.norm,
                norm=get_norm(self.norm, conv_dims),
                activation=F.relu,
            )
            self.add_module("amodal_fcn{}".format(k + 1), conv)
            self.amodal_conv_norm_relus.append(conv)

        self.inmodal_conv_norm_relus = []
        for k in range(num_conv):
            conv = Conv2d(
                768 if k == 0 else conv_dims,
                conv_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not self.norm,
                norm=get_norm(self.norm, conv_dims),
                activation=F.relu,
            )
            self.add_module("inmodal_fcn{}".format(k + 1), conv)
            self.inmodal_conv_norm_relus.append(conv)

        self.amodal_2_conv_norm_relus = []
        for k in range(num_conv):
            conv = Conv2d(
                768 if k == 0 else conv_dims,
                conv_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not self.norm,
                norm=get_norm(self.norm, conv_dims),
                activation=F.relu,
            )
            self.add_module("amodal_fcn_2{}".format(k + 1), conv)
            self.amodal_2_conv_norm_relus.append(conv)

        self.occluder_conv_norm_relus = []
        for k in range(num_conv):
            conv = Conv2d(
                256 if k == 0 else conv_dims,
                conv_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not self.norm,
                norm=get_norm(self.norm, conv_dims),
                activation=F.relu,
            )
            self.add_module("occluder_fcn{}".format(k + 1), conv)
            self.occluder_conv_norm_relus.append(conv)

        self.invisible_conv_norm_relus = []
        for k in range(num_conv):
            conv = Conv2d(
                768 if k == 0 else conv_dims,
                conv_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not self.norm,
                norm=get_norm(self.norm, conv_dims),
                activation=F.relu,
            )
            self.add_module("invisible_fcn{}".format(k + 1), conv)
            self.invisible_conv_norm_relus.append(conv)

        self.visible_deconv = ConvTranspose2d(
            conv_dims if num_conv > 0 else input_channels,
            conv_dims,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        self.amodal_deconv = ConvTranspose2d(
            conv_dims if num_conv > 0 else input_channels,
            conv_dims,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        self.amodal_2_deconv = ConvTranspose2d(
            conv_dims if num_conv > 0 else input_channels,
            conv_dims,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        self.inmodal_deconv = ConvTranspose2d(
            conv_dims if num_conv > 0 else input_channels,
            conv_dims,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        self.occluder_deconv = ConvTranspose2d(
            conv_dims if num_conv > 0 else input_channels,
            conv_dims,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        self.invisible_deconv = ConvTranspose2d(
            conv_dims if num_conv > 0 else input_channels,
            conv_dims,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        self.amodal_gcn  = GCN_Module( input_channels = conv_dims, dropout_rate = 0.1)

        num_mask_classes = 1 if cls_agnostic_mask else num_classes
        self.visible_predictor = Conv2d(conv_dims, num_mask_classes, kernel_size=1, stride=1, padding=0)
        self.amodal_predictor = Conv2d(conv_dims, num_mask_classes, kernel_size=1, stride=1, padding=0)
        self.inmodal_predictor = Conv2d(conv_dims, num_mask_classes, kernel_size=1, stride=1, padding=0)
        self.occluder_predictor = Conv2d(conv_dims, num_mask_classes, kernel_size=1, stride=1, padding=0)
        self.invisible_predictor = Conv2d(conv_dims, num_mask_classes, kernel_size=1, stride=1, padding=0)
        self.amodal_2_predictor = Conv2d(conv_dims, num_mask_classes, kernel_size=1, stride=1, padding=0)

        for layer in self.visible_conv_norm_relus:
            weight_init.c2_msra_fill(layer)

        for layer in self.amodal_conv_norm_relus:
            weight_init.c2_msra_fill(layer)

        for layer in self.inmodal_conv_norm_relus:
            weight_init.c2_msra_fill(layer)

        for layer in self.occluder_conv_norm_relus:
            weight_init.c2_msra_fill(layer)

        for layer in self.invisible_conv_norm_relus:
            weight_init.c2_msra_fill(layer)

        for layer in self.amodal_2_conv_norm_relus:
            weight_init.c2_msra_fill(layer)

        for layer in [self.invisible_deconv, self.amodal_2_deconv, self.visible_deconv, self.amodal_deconv, self.inmodal_deconv, self.occluder_deconv]:
            weight_init.c2_msra_fill(layer)

        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.visible_predictor.weight, std=0.001)
        if self.visible_predictor.bias is not None:
            nn.init.constant_(self.visible_predictor.bias, 0)

        nn.init.normal_(self.amodal_predictor.weight, std=0.001)
        if self.amodal_predictor.bias is not None:
            nn.init.constant_(self.amodal_predictor.bias, 0)

        nn.init.normal_(self.inmodal_predictor.weight, std=0.001)
        if self.inmodal_predictor.bias is not None:
            nn.init.constant_(self.inmodal_predictor.bias, 0)

        nn.init.normal_(self.occluder_predictor.weight, std=0.001)
        if self.occluder_predictor.bias is not None:
            nn.init.constant_(self.occluder_predictor.bias, 0)

        nn.init.normal_(self.invisible_predictor.weight, std=0.001)
        if self.invisible_predictor.bias is not None:
            nn.init.constant_(self.invisible_predictor.bias, 0)

        nn.init.normal_(self.amodal_2_predictor.weight, std=0.001)
        if self.amodal_2_predictor.bias is not None:
            nn.init.constant_(self.amodal_2_predictor.bias, 0)

    def forward(self, x, c_iter ):

        x_ori = x.clone()

        # ##############开始occluder branch
        # xori+xbox --> occluder
        for cnt, layer in enumerate(self.occluder_conv_norm_relus):
            x = layer(x)

            # occluder_predictor_input
        occluder_predictor_features = x.clone()
            # occluder
        oc_features = x.clone()


        # ##############开始visible branch
        # occluder+xori+xbox --> visible
        x = torch.cat([x, x_ori], 1)   
        for cnt, layer in enumerate(self.visible_conv_norm_relus):
            x = layer(x)

            # visible_predictor_input
        vi_predictor_features = x.clone()
            # visible
        vi_features = x.clone()


        # ##############开始invisible branch
        # occluder+xori+visible --> invisible
        x = torch.cat([x, x_ori, oc_features], 1)   
        for cnt, layer in enumerate(self.invisible_conv_norm_relus):
            x = layer(x)

            # invisible_predictor_input
        invi_predictor_features = x.clone()
            # invisible
        invi_features = x.clone()


        # #############开始amodal branch
        # occluder + xori + visible + invisible --> amodal
        x = torch.cat([x, x_ori, oc_features, vi_features], 1)         
        for cnt, layer in enumerate(self.amodal_conv_norm_relus):
            x = layer(x)
            # 先两个conv2d，然后gcn，然后再两个conv2d
            if cnt == 3 and len(x) != 0:
                x = self.amodal_gcn(x)

            # amodal_predictor_input
        amodal_predictor_features = x.clone()
            # amodal
        amodal_features = x.clone()

        # #############开始inmodal branch
        # amodal + visible --> inmodal
        x = torch.cat([x, vi_features, x_ori], 1)         
        for cnt, layer in enumerate(self.inmodal_conv_norm_relus):
            x = layer(x)

            # amodal_predictor_input
        inmodal_predictor_features = x.clone()

        # #############开始amodal 2 branch
        # amodal + visible --> amodal 2
        x = torch.cat([x, amodal_features, x_ori], 1)         
        for cnt, layer in enumerate(self.amodal_2_conv_norm_relus):
            x = layer(x)

        # occluder mask
        occluder_predictor_features = F.relu(self.occluder_deconv(occluder_predictor_features))
        occluder_mask = self.occluder_predictor(occluder_predictor_features) 

        # visible mask
        vi_predictor_features = F.relu(self.visible_deconv(vi_predictor_features))
        visible_mask = self.visible_predictor(vi_predictor_features) 

        # invisible mask
        invi_predictor_features = F.relu(self.invisible_deconv(invi_predictor_features))
        invisible_mask = self.invisible_predictor(invi_predictor_features) 

        # amodal mask
        amodal_predictor_features = F.relu(self.amodal_deconv(amodal_predictor_features))
        amodal_mask = self.amodal_predictor(amodal_predictor_features) 

        # inmodal mask
        inmodal_predictor_features = F.relu(self.inmodal_deconv(inmodal_predictor_features))
        inmodal_mask = self.inmodal_predictor(inmodal_predictor_features)  

        # amodal_2 mask
        x = F.relu(self.amodal_2_deconv(x))
        amodal_2_mask = self.amodal_2_predictor(x)   

        return visible_mask, amodal_mask, occluder_mask, invisible_mask, inmodal_mask, amodal_2_mask

class GCN_Module(nn.Module):
    def __init__(self, input_channels = 256, dropout_rate = 0.1):
        super().__init__()
        self.query_transform = Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.key_transform = Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.value_transform = Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.output_transform = Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.scale = 1.0 / (input_channels ** 0.5)
        self.blocker = nn.LayerNorm(input_channels) # should be zero initialized

        self.attn_dropout = nn.Dropout(dropout_rate)
        self.out_dropout = nn.Dropout(dropout_rate)

        for layer in [self.query_transform,self.key_transform, self.value_transform, self.output_transform]:
            weight_init.c2_msra_fill(layer)

    def forward(self, x):
        B, C, H, W = x.size()
        # 首先构建gcn的 query, key, value
        # x: B,C,H,W
        # x_query: B,C,HW
        #x_input = AddCoords()(x)
        x_query = self.query_transform(x).view(B, C, -1)
        # x_query: B,HW,C
        x_query = torch.transpose(x_query, 1, 2)
        # x_key: B,C,HW
        x_key = self.key_transform(x).view(B, C, -1)
        # x_value: B,C,HW
        x_value = self.value_transform(x).view(B, C, -1)
        # x_value: B,HW,C
        x_value = torch.transpose(x_value, 1, 2)

        # query * key 构建相似度矩阵
        # W = Q^T K: B,HW,HW
        x_w = torch.matmul(x_query, x_key) * self.scale

        x_w = F.softmax(x_w, dim=-1)

        # attn_dropout
        x_w = self.attn_dropout(x_w)

        # 相似度矩阵 * value 构建关系矩阵
        # x_relation = WV: B,HW,C
        x_relation = torch.matmul(x_w, x_value)
        # x_relation = B,C,HW
        x_relation = torch.transpose(x_relation, 1, 2)
        # x_relation = B,C,H,W
        x_relation = x_relation.view(B,C,H,W)

        # output_transform B,HW,C
        x_relation = self.output_transform(x_relation).view(B,C,-1).transpose(1, 2)

        # out_dropout B,HW,C
        x_relation = self.out_dropout(x_relation)       

        # 加上原来的x : feature maps, 输出至x  B,HW,C
        x = x.view(B,C,-1).transpose(1, 2) + x_relation

        # layernorm  B,C,HW
        x = self.blocker(x).transpose(1, 2).view(B,C,H,W)

        return x

def build_mask_head(cfg, input_shape):
    """
    Build a mask head defined by `cfg.MODEL.ROI_MASK_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_MASK_HEAD.NAME
    return ROI_MASK_HEAD_REGISTRY.get(name)(cfg, input_shape)
