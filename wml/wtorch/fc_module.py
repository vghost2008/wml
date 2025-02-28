import warnings
import copy
import torch.nn as nn
import wml.wtorch.nn as wnn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm
from .nn_utils import fuse_linear_bn_eval


Linear = nn.Linear
class FCModule(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias='auto',
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 inplace=True,
                 ):
        super().__init__()
        self.is_fused = False
        norm_cfg = copy.deepcopy(norm_cfg)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = not self.with_norm
        self.with_bias = bias

        modules = []
        # build convolution layer
        fc = nn.Linear(
            in_channels,
            out_channels,
            bias=bias)
        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = in_channels
        self.out_channels = out_channels
        modules.append(fc)

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            norm_type = norm_cfg.pop('type')
            norm = wnn.get_norm1d(norm_type,
                                  self.out_channels,
                                  norm_args=norm_cfg)
            if self.with_bias:
                if isinstance(norm, (_BatchNorm, _InstanceNorm)):
                    warnings.warn(
                        'Unnecessary conv bias before batch/instance norm')
            modules.append(norm)

        # build activation layer
        if self.with_activation:
            act_cfg_ = act_cfg.copy()
            # nn.Tanh has no 'inplace' argument
            if act_cfg_['type'] not in [
                    'Tanh', 'PReLU', 'Sigmoid', 'HSigmoid', 'Swish', 'GELU'
            ]:
                act_cfg_.setdefault('inplace', inplace)
            act_type = act_cfg_.pop('type')
            inplace = act_cfg_.pop('inplace')
            activate = wnn.get_activation(act_type,inplace)
            modules.append(activate)
        self.fc = nn.Sequential(*modules)

    def fuse(self):
        if self.training:
            print(f"ERROR: fuse training FCModule, skip operation")
            return
        if self.is_fused:
            return
        if self.with_norm and isinstance(self.fc[1],nn.BatchNorm1d):
            self.fc[0] = fuse_linear_bn_eval(self.fc[0],self.fc[1])
            del self.fc[1]
        self.is_fused = True

    def forward(self, x):
        return self.fc.forward(x)