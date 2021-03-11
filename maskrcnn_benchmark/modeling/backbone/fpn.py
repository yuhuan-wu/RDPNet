# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn
from ..make_layers import make_conv3x3, make_conv1x1



class FPN(nn.Module):
    """
    Module that adds FPN on top of a list of feature maps.
    The feature maps are currently supposed to be in increasing depth
    order, and must be consecutive
    """

    def __init__(
            self, in_channels_list, out_channels, conv_block, top_blocks=None, dense=False, att_dense=False
    ):
        """
        Arguments:
            in_channels_list (list[int]): number of channels for each feature map that
                will be fed
            out_channels (int): number of channels of the FPN representation
            top_blocks (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list
        """
        super(FPN, self).__init__()
        self.inner_blocks = []
        self.layer_blocks = []
        self.top_down_dense = False
        self.att_dense = att_dense
        self.dense = dense
        self.dense_down_blocks = []
        for idx, in_channels in enumerate(in_channels_list, 1):
            inner_block = "fpn_inner{}".format(idx)
            layer_block = "fpn_layer{}".format(idx)

            if in_channels == 0:
                continue
            inner_block_module = conv_block(in_channels, out_channels, 1)

            if self.top_down_dense:
                # top down dense does not work!
                if idx == 1:
                    layer_block_module = conv_block(out_channels, out_channels, 3, 1)
                else:
                    layer_block_module = nn.Sequential(
                        make_conv1x1(out_channels * (len(in_channels_list) - idx + 1), out_channels),
                        conv_block(out_channels, out_channels, 3, 1)
                    )
                    att_block_module = make_conv3x3(out_channels, out_channels, use_gn=False, use_relu=False)
                    att_block = "att_block{}".format(idx)
                    self.add_module(att_block, att_block_module)
                    self.att_blocks.append(att_block)
            else:
                layer_block_module = conv_block(out_channels, out_channels, 3, 1)

            self.add_module(inner_block, inner_block_module)
            self.add_module(layer_block, layer_block_module)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)

        REFINE = False
        self.refine_blocks = []
        self.att_blocks = []
        """
        WARNING 
        we have used this value [self.att_blocks].
        So we could not enable both top-down and bottom-up attention.
        """
        if att_dense or dense:
            for idx, in_channels in enumerate(range(5), 1):
                refine_block = "refine_block_{}".format(idx)
                if idx == 1:
                    module = make_conv3x3(out_channels, out_channels,
                                          use_gn=False)  
                else:
                    module = nn.Sequential(
                        make_conv1x1(out_channels * idx, out_channels, use_gn=False),
                        make_conv3x3(out_channels, out_channels, use_gn=False)
                    )

                self.add_module(refine_block, module)
                self.refine_blocks.append(refine_block)
            if att_dense:
                for idx, in_channels in enumerate(range(4), 1):
                    module = make_conv3x3(out_channels, out_channels, use_gn=False, use_relu=False)
                    att_block = "att_block_{}".format(idx)
                    self.add_module(att_block, module)
                    self.att_blocks.append(att_block)

        self.top_blocks = top_blocks

    def forward(self, x, test=False):
        """
        Arguments:
            x (list[Tensor]): feature maps for each feature level.
        Returns:
            results (tuple[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        results = []
        for feature, inner_block in zip(x[::-1], self.inner_blocks[::-1]):
            results.append(
                getattr(self, inner_block)(feature)
            )
        last_inner = getattr(self, self.inner_blocks[-1])(x[-1])
        results = []
        results_CX = []
        results.append(getattr(self, self.layer_blocks[-1])(last_inner))

        if not self.top_down_dense:
            # run top-down FPN
            for feature, inner_block, layer_block in zip(
                    x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]
            ):
                if not inner_block:
                    continue

                inner_top_down = F.interpolate(last_inner, scale_factor=2, mode="nearest")
                inner_lateral = getattr(self, inner_block)(feature)
                # TODO use size instead of scale to make it robust to different sizes
                # inner_top_down = F.upsample(last_inner, size=inner_lateral.shape[-2:],
                # mode='bilinear', align_corners=False)
                last_inner = inner_lateral + inner_top_down
                results.insert(0, getattr(self, layer_block)(last_inner))
            if isinstance(self.top_blocks, LastLevelP6P7) or isinstance(self.top_blocks, LastLevelP6):
                last_results = self.top_blocks(x[-1], results[-1])
                results.extend(last_results)
            elif isinstance(self.top_blocks, LastLevelMaxPool):
                last_results = self.top_blocks(results[-1])
                results.extend(last_results)
        else:
            # run top-down dense FPN
            for feature, inner_block, layer_block, att_block in zip(
                    x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1], self.att_blocks
            ):
                if not inner_block:
                    continue

                new_results_up = []
                inner_lateral = getattr(self, inner_block)(feature)
                attention = getattr(self, att_block)(inner_lateral)
                for j in range(len(results)):
                    new_results_up.append(
                        attention * F.interpolate(results[j], size=feature.shape[2:], mode='bilinear',
                                                  align_corners=False)
                    )

                results.append(
                    getattr(self, layer_block)(
                        torch.cat(
                            (torch.cat(new_results_up, dim=1), inner_lateral), dim=1
                        )
                    )
                )
            if isinstance(self.top_blocks, LastLevelP6P7) or isinstance(self.top_blocks, LastLevelP6):
                last_results = self.top_blocks(x[-1], results[-1])
                results.extend(last_results)
            elif isinstance(self.top_blocks, LastLevelMaxPool):
                last_results = self.top_blocks(results[-1])
                results.extend(last_results)

        # run bottom up (regularized) dense
        if len(self.refine_blocks):
            new_results = []
            for i in range(len(results)):
                if i == 0:
                    new_results.append(getattr(self, self.refine_blocks[i])(results[i]))
                    cat_result = new_results[0]
                else:
                    new_results_down = []
                    if self.att_dense:
                        # run regularized
                        attention = getattr(self, self.att_blocks[i - 1])(results[i])
                        attention = torch.sigmoid(attention)
                        for j in range(i):
                            new_results_down.append(
                                attention * F.interpolate(new_results[j], size=results[i].shape[2:], mode='bilinear',
                                                          align_corners=None)
                            )
                    else:
                        # run directly
                        for j in range(i):
                            new_results_down.append(
                                F.interpolate(new_results[j], size=results[i].shape[2:], mode='bilinear',
                                              align_corners=None)
                            )
                    new_results.append(getattr(self, self.refine_blocks[i])(
                        torch.cat(
                            (torch.cat(new_results_down, 1), results[i]), 1
                        )
                    ))
            results = new_results
        return tuple(results)


class LastLevelMaxPool(nn.Module):
    def forward(self, x):
        return [F.max_pool2d(x, 1, 2, 0)]


class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7.
    """

    def __init__(self, in_channels, out_channels):
        super(LastLevelP6P7, self).__init__()
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            nn.init.kaiming_uniform_(module.weight, a=1)
            nn.init.constant_(module.bias, 0)
        self.use_P5 = in_channels == out_channels

    def forward(self, c5, p5):
        x = p5 if self.use_P5 else c5
        p6 = self.p6(x)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]


class LastLevelP6(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7.
    """

    def __init__(self, in_channels, out_channels):
        super(LastLevelP6, self).__init__()
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        for module in [self.p6]:
            nn.init.kaiming_uniform_(module.weight, a=1)
            nn.init.constant_(module.bias, 0)
        self.use_P5 = in_channels == out_channels

    def forward(self, c5, p5):
        x = p5 if self.use_P5 else c5
        p6 = F.relu(self.p6(x))
        return [p6]