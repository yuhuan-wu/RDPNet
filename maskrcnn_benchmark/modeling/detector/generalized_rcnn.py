# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn
import time
from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
from ..backbone.segmentation import SegHead
import os
import numpy as np


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        if cfg.MODEL.SEG_ON:
            self.seg_heads = SegHead(cfg, self.backbone.out_channels)
            #self.fpn_enhance = FPN_ENHANCE(self.backbone.out_channels)
        else:
            self.seg_heads = None
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)
        self.proposals_logger = open('proposal_log.txt', 'w') if os.path.exists('proposal_log.txt') else open('proposal_log.txt', 'a')
        print(sum([p.numel() for p in self.parameters()]))

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        if self.seg_heads is not None:
            top_features = features[0]#.detach()  # DONE: do not use detach()! WYH
            masks, seg_losses = self.seg_heads(top_features, targets)
            torch.cuda.synchronize()
            t3 = time.time() - start - t1 - t2
        else:
            masks = None; seg_losses = None

        proposals, proposal_losses = self.rpn(images,  features, targets, masks=masks)

        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, masks=masks, targets=targets)
        else:
            x = features
            result = proposals
            detector_losses = {}
        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            if seg_losses is not None:
                losses.update(seg_losses)
            return losses
        return result
