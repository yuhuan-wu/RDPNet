# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import nn
from torch.nn import functional as F
import torch
from ..box_head.roi_box_feature_extractors import ResNet50Conv5ROIFeatureExtractor
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.modeling.make_layers import make_conv3x3, make_conv1x1


registry.ROI_MASK_FEATURE_EXTRACTORS.register(
    "ResNet50Conv5ROIFeatureExtractor", ResNet50Conv5ROIFeatureExtractor
)

@registry.ROI_MASK_FEATURE_EXTRACTORS.register("InstanceSaliencyMaskDecoder")
class MaskDecoder(nn.Module):

    def __init__(self, cfg, in_channels):
        super(MaskDecoder, self).__init__()
        sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        scales = cfg.MODEL.ROI_MASK_HEAD.POOLER_SCALES
        resolutions = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION #[32, 16, 8, 4, 4]
        poolers = []
        for idx, resolution in enumerate(resolutions):
            poolers.append(
                Pooler(
                    output_size=(resolution, resolution),
                    scales=[scales[idx]],
                    sampling_ratio=sampling_ratio,
                )
            )
        self.poolers = poolers

        inner_blocks = []
        for idx in range(len(scales)):
            inner_block = make_conv1x1(in_channels, 128, dilation=1, use_gn=1, use_relu=1)
            block_name = 'inner_maskdecoder_{}'.format(idx+1)
            self.add_module(block_name, inner_block)
            inner_blocks.append(block_name)

        conv_blocks = []
        for idx in range(len(scales)-1):
            if idx < len(scales) - 1:
                conv_block = make_conv3x3(128, 128, use_gn=1, use_relu=1)
            else:
                conv_block = nn.Sequential(
                    make_conv3x3(128, 128, use_gn=1, use_relu=1),
                    make_conv3x3(128, 128, use_gn=1, use_relu=1),
                )
            block_name = 'conv_maskdecoder_{}'.format(idx + 1)
            self.add_module(block_name, conv_block)
            conv_blocks.append(block_name)

        self.inner_blocks = inner_blocks
        self.conv_blocks = conv_blocks
        self.out_channels = 128

    def forward(self, x, masks=None, proposals=None):
        features = []
        for i in range(len(x)):
            features.append(getattr(self, self.inner_blocks[i])(x[i]))
            #print(features[i].shape)
        last_feature = self.poolers[4]([features[-1]], proposals) + self.poolers[3]([features[-2]], proposals)
        last_feature = getattr(self, self.conv_blocks[0])(last_feature)
        for (pooler, feature, conv_block) in zip(
            self.poolers[:3][::-1], features[:3][::-1], self.conv_blocks[1:]
        ):
            feature_lateral = pooler([feature], proposals)
            feature_top_down = F.interpolate(
                last_feature, size=feature_lateral.shape[2:], mode='bilinear', align_corners=False,
            )
            last_feature = getattr(self, conv_block)(feature_top_down + feature_lateral)
        return last_feature







@registry.ROI_MASK_FEATURE_EXTRACTORS.register("MaskRCNNFPNFeatureExtractor")
class MaskRCNNFPNFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels):
        """
        Arguments:
            num_classes (int): number of output classes
            input_size (int): number of channels of the input once it's flattened
            representation_size (int): size of the intermediate representation
        """
        super(MaskRCNNFPNFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION[0]
        scales = cfg.MODEL.ROI_MASK_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        pooler_mask = Pooler(
            output_size=(resolution, resolution),
            scales=(1., ),
            sampling_ratio=sampling_ratio,
        )
        input_size = in_channels
        self.pooler = pooler
        self.pooler_mask = pooler_mask

        use_gn = cfg.MODEL.ROI_MASK_HEAD.USE_GN
        layers = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS
        dilation = cfg.MODEL.ROI_MASK_HEAD.DILATION

        next_feature = input_size
        self.blocks = []
        for layer_idx, layer_features in enumerate(layers, 1):
            layer_name = "mask_fcn{}".format(layer_idx)
            flag = cfg.MODEL.SEG_ON_ADD_CHANEL and (layer_idx == 1)
            module = make_conv3x3(
                next_feature + flag, layer_features,
                dilation=dilation, stride=1, use_gn=use_gn
            )
            self.add_module(layer_name, module)
            next_feature = layer_features
            self.blocks.append(layer_name)
        self.out_channels = layer_features


    def forward(self, x, masks=None, proposals=None):
        x = self.pooler(x, proposals)
        if masks is not None:
            rois = self.pooler_mask([masks], proposals)
            #rois = (rois - 0.5) * 2
            x = rois * x
            #x = torch.cat((x, rois),dim=1)
        for layer_name in self.blocks:
            x = F.relu(getattr(self, layer_name)(x))

        return x


    def forward_default(self, x, proposals):
        x = self.pooler(x, proposals)

        for layer_name in self.blocks:
            x = F.relu(getattr(self, layer_name)(x))

        return x


def make_roi_mask_feature_extractor(cfg, in_channels):
    func = registry.ROI_MASK_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg, in_channels)
