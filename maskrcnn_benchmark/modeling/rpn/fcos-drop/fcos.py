import math
import torch
import torch.nn.functional as F
from torch import nn

from .inference import make_fcos_postprocessor
from .loss import make_fcos_loss_evaluator

from maskrcnn_benchmark.layers import Scale


class FCOSHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.MODEL.FCOS.NUM_CLASSES - 1

        cls_tower = []
        bbox_tower = []
        eval_tower = []
        cls_tower.append(
            nn.Conv2d(
                in_channels+cfg.MODEL.SEG_ON,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )
        cls_tower.append(nn.GroupNorm(32, in_channels))
        cls_tower.append(nn.ReLU())
        bbox_tower.append(
            nn.Conv2d(
                in_channels+cfg.MODEL.SEG_ON,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )
        bbox_tower.append(nn.GroupNorm(32, in_channels))
        bbox_tower.append(nn.ReLU())
        for i in range(cfg.MODEL.FCOS.NUM_CONVS - 1):
            cls_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())
            if cfg.MODEL.FCOS.EVAL_HEAD:
                eval_tower.append(nn.GroupNorm(32, in_channels))
                eval_tower.append(
                    nn.Conv2d(
                        in_channels,
                        in_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1
                    )
                )
                eval_tower.append(nn.GroupNorm(32, in_channels))
                eval_tower.append(nn.ReLU())
            else:
                self.eval_tower = None


        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1,
            padding=1
        )
        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )
        if cfg.MODEL.FCOS.EVAL_HEAD:
            self.add_module('eval_tower', nn.Sequential(*eval_tower))
            self.eval_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1, padding=1,
            )
            for modules in [self.eval_tower, self.eval_pred]:
                for l in modules.modules():
                    if isinstance(l, nn.Conv2d):
                        torch.nn.init.normal_(l.weight, std=0.01)
                        torch.nn.init.constant_(l.bias, 0)

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(6)])

    def forward(self, x, masks=None):
        logits = []
        bbox_reg = []
        centerness = []
        bbox_eval = []

        for l, feature in enumerate(x):
            if masks is not None:
                new_mask = F.interpolate(masks, size=feature.shape[2:4], mode='bilinear')
                new_feature = torch.cat((new_mask, feature), 1)
            else:
                new_feature = feature
            cls_tower = self.cls_tower(new_feature)
            bbox_tower = self.bbox_tower(new_feature)
            logits.append(self.cls_logits(cls_tower))
            centerness.append(F.interpolate(masks, size=feature.shape[2:], mode='bilinear'))#self.centerness(cls_tower))
            bbox_reg.append(torch.exp(self.scales[l](
                self.bbox_pred(bbox_tower)
            )))
            if self.eval_tower is not None:
                bbox_eval.append(self.eval_tower(new_feature))
        return logits, bbox_reg, centerness, bbox_eval


class FCOSModule(torch.nn.Module):
    """
    Module for FCOS computation. Takes feature maps from the backbone and
    FCOS outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(FCOSModule, self).__init__()

        head = FCOSHead(cfg, in_channels)

        box_selector_test = make_fcos_postprocessor(cfg)
        box_selector_train = make_fcos_postprocessor(cfg, train=True)
        loss_evaluator = make_fcos_loss_evaluator(cfg)
        self.head = head
        self.box_selector_train = box_selector_train
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES

    def forward(self, images, features, targets=None, masks=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        box_cls, box_regression, centerness, box_eval = self.head(features, masks=masks)
        locations = self.compute_locations(features)
 
        if self.training:
            return self._forward_train(
                locations, box_cls, 
                box_regression, 
                centerness, targets, images.image_sizes,
            )
        else:
            return self._forward_test(
                locations, box_cls, box_regression, 
                centerness, images.image_sizes,
            )

    def _forward_train(self, locations, box_cls, box_regression, centerness, targets, image_sizes):
        loss_box_cls, loss_box_reg = self.loss_evaluator(
            locations, box_cls, box_regression, centerness, targets,
        )
        with torch.no_grad():
            boxes = self.box_selector_train(locations, box_cls, box_regression,
            centerness, image_sizes, targets=targets,
            )
        losses = {
            "loss_cls": loss_box_cls,
            "loss_reg": loss_box_reg,
            #"loss_centerness": loss_centerness
        }
        return boxes, losses

    def _forward_test(self, locations, box_cls, box_regression, centerness, image_sizes):
        boxes = self.box_selector_test(
            locations, box_cls, box_regression, 
            centerness, image_sizes,
        )
        return boxes, {}

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations

def build_fcos(cfg, in_channels):
    return FCOSModule(cfg, in_channels)
