import torch
import torch.nn.functional as F
from torch import nn
from maskrcnn_benchmark.modeling.make_layers import make_conv3x3, make_conv1x1
from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.layers import ConvTranspose2d
from maskrcnn_benchmark.layers import DeformConvPack
import numpy as np
import cv2

class SegmentationBranch(nn.Module):
    def __init__(self, cfg, in_channels):
        super(SegmentationBranch, self).__init__()
        self.in_channels = in_channels
        self.seg_fcn1 = make_conv1x1(in_channels, in_channels, use_relu=0, kaiming_init=False)
        self.seg_fcn2 = make_conv3x3(in_channels, in_channels, use_gn=1, use_relu=1)
        self.seg_fcn3 = make_conv3x3(in_channels, in_channels, use_gn=1, use_relu=1)
        self.seg_fcn4 = make_conv3x3(in_channels, in_channels, use_gn=1, use_relu=1)
        self.predict = make_conv1x1(in_channels, 1, kaiming_init=False)

    def forward(self, x):
        #x1 = F.interpolate(x[1], scale_factor=2, mode='bilinear')
        #x2 = F.interpolate(x[2], scale_factor=4, mode='bilinear')
        #x = x[0] + x1 + x2
        x = self.seg_fcn1(x)
        x = self.seg_fcn2(x)
        x = self.seg_fcn3(x)
        x = self.seg_fcn4(x)
        #x = F.relu_(self.dcn_branch(x))
        x = F.interpolate(self.predict(x), scale_factor=8, mode='bilinear')
        return torch.sigmoid(x)



class SegmentationLoss(object):
    def __init__(self, cfg):
        super(SegmentationLoss, self).__init__()
        self.weight = 1. #cfg.SEGLOSS.WEIGHT

    def mask_decode(self, size, target):
        segmentation_masks = target.get_field("masks")
        mask = np.zeros(size)
        instances = segmentation_masks.get_mask_tensor().data.cpu().numpy().astype(np.uint8)

        if len(instances.shape) == 2:
            mask = cv2.resize(instances, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
        else:
            try:
                instances = cv2.resize(instances.transpose(1,2,0), (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
            except IndexError:
                print("error size: {}".format(size))
            instances = instances.transpose(2, 0, 1)
            for instance in instances:
                mask += instance
            mask = mask >= 1.
        return mask.astype(np.float32)

    def __call__(self, features, targets):
        masks = []
        for feature, target in zip(features, targets):
            mask = self.mask_decode(feature.shape[-2:], target)
            masks.append(mask)
        masks = np.array(masks).astype(np.float32)
        masks = torch.from_numpy(masks).cuda().unsqueeze(dim=1)
        if targets is not None:
            mask_loss = self.weight * F.binary_cross_entropy(features, masks)
            mask_loss /= len(targets)
        else:
            mask_loss = None
        return mask_loss, masks


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.normal_(0, 0.01)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SegHead(nn.Module):

    def __init__(self, cfg, in_channels):
        super(SegHead, self).__init__()
        self.seg_branch = SegmentationBranch(cfg, in_channels)
        self.seg_loss = SegmentationLoss(cfg)



    def forward(self, features, targets=None):
        out = self.seg_branch(features)
        if targets is not None:
            loss_seg, _ = self.seg_loss(out, targets)
            losses = {"loss_seg": loss_seg}
            return out, losses
        else:
            return out, {}
