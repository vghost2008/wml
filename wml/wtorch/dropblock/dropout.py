import torch
import torch.nn.functional as F
from torch import nn


class WDropout(nn.Module):
    r"""Randomly zeroes 2D spatial blocks of the input tensor.

    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.

    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop

    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`

    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890

    """

    def __init__(self, drop_prob,inplace=False):
        super().__init__()

        self.drop_prob = drop_prob
        self.inplace = inplace
        self.cur_drop_prob = 0.0

    def forward(self, x):
        if not self.training or self.drop_prob <= 0.:
            self.cur_drop_prob = 0.0
            return x
        else:
            self.cur_drop_prob = self.drop_prob
            return F.dropout(x, self.drop_prob, self.training, self.inplace)
