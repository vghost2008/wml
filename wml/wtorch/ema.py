#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import math
from copy import deepcopy

import torch
import torch.nn as nn

__all__ = ["ModelEMA", "is_parallel"]


def is_parallel(model):
    """check if model is in parallel mode."""
    parallel_type = (
        nn.parallel.DataParallel,
        nn.parallel.DistributedDataParallel,
    )
    return isinstance(model, parallel_type)


class ModelEMA:
    """
    Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0, base_updates=2000):
        """
        Args:
            model (nn.Module): model to apply EMA.
            decay (float): ema decay reate.
            updates (int): counter of EMA updates.
        """
        # Create EMA(FP32)
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()
        self.updates = updates
        # decay exponential ramp (to help early epochs)
        self.base_updates = base_updates
        self.base_decay = decay
        self.decay = lambda x: decay * (1 - math.exp(-x / self.base_updates))
        self.ema_persent = 0.0
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        #print(f"EMA Update.")
        with torch.no_grad():
            d = self.decay(self.updates) #old persent
            self.ema_persent = d
            self.updates += 1

            msd = (
                model.module.state_dict() if is_parallel(model) else model.state_dict()
            )  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1.0 - d) * msd[k].detach()

    
    def __repr__(self) -> str:
        return f"EMA: (base_updates={self.base_updates}, decay={self.base_decay})"
