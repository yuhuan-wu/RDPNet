# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch


# TODO maybe push this to nn?
def smooth_l1_loss(input, target, beta=1. / 9, size_average=True, weight=None):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    input = torch.log(input)
    target = torch.log(target)
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if weight is not None and weight.sum() > 0:
        return (loss * weight).sum() / weight.sum()
    if size_average:
        return loss.mean()
    return loss.sum()
