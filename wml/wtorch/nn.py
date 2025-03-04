import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv_ws import ConvWS2d
from collections.abc import Iterable
from torch.nn import Parameter
import math
from collections import OrderedDict
from torch import Tensor
#from einops import rearrange


def _clone_tensors(x):
    if isinstance(x,(list,tuple)):
        return [v.clone() for v in x]
    return x.clone()


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            if len(x.shape)==4:
                x = self.weight[:, None, None] * x + self.bias[:, None, None]
            else:
                x = self.weight[:, None] * x + self.bias[:, None]
            return x

class Identity(nn.Module):
    def __init__(self,name="Identity"):
        self.name = name
        self.cache = None
        self.grad_input = None
        self.grad_output = None
        super().__init__()
        self.register_backward_hook(self.backward_hook)


    def backward_hook(self,model,grad_input,grad_output):
        self.grad_input = _clone_tensors(grad_input)
        self.grad_output = _clone_tensors(grad_output)

    def forward(self,x):
        self.cache = x
        return x.clone()

    def __repr__(self):
        return self.name

class LayerNorm2d(nn.LayerNorm):
    """LayerNorm on channels for 2d images.

    Args:
        num_channels (int): The number of channels of the input tensor.
        eps (float): a value added to the denominator for numerical stability.
            Defaults to 1e-5.
        elementwise_affine (bool): a boolean value that when set to ``True``,
            this module has learnable per-element affine parameters initialized
            to ones (for weights) and zeros (for biases). Defaults to True.
    """

    def __init__(self, num_features: int, **kwargs) -> None:
        super().__init__(num_features, **kwargs)
        self.num_channels = self.normalized_shape[0]


    def forward(self, x, data_format='channel_first'):
        """Forward method.

        Args:
            x (torch.Tensor): The input tensor.
            data_format (str): The format of the input tensor. If
                ``"channel_first"``, the shape of the input tensor should be
                (B, C, H, W). If ``"channel_last"``, the shape of the input
                tensor should be (B, H, W, C). Defaults to "channel_first".
        """
        assert x.dim() == 4, 'LayerNorm2d only supports inputs with shape ' \
            f'(N, C, H, W), but got tensor with shape {x.shape}'
        if data_format == 'channel_first':
            x = x.permute(0, 2, 3, 1)
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias,
                             self.eps)
            # If the output is discontiguous, it may cause some unexpected
            # problem in the downstream tasks
            x = x.permute(0, 3, 1, 2).contiguous()
        elif data_format == 'channel_last':
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias,
                             self.eps)
        return x


class EvoNormS0(nn.Module):
    def __init__(self, num_groups,num_features, eps=1e-6, scale=True):
        super().__init__()
        self.num_groups = num_groups
        self.num_features = num_features
        if scale:
            self.gamma = nn.Parameter(torch.ones([1,num_groups,num_features//num_groups,1,1]))
        self.beta = nn.Parameter(torch.zeros([1,num_groups,num_features//num_groups,1,1]))
        self.v1 = nn.Parameter(torch.ones([1,num_groups,num_features//num_groups,1,1]))
        self.eps = eps
        self.scale = scale
    
    def forward(self, x):
        N,C,H,W = x.shape
        G = self.num_groups
        x = x.view([N,G,C//G,H,W])
        var = x.std(dim=(2,3,4),keepdim=True)
        gain = torch.rsqrt(var+self.eps)
        if self.scale:
            gain = gain*self.gamma
        
        x = x*torch.sigmoid(x*self.v1)*gain+self.beta

        return x.view([N,C,H,W]).contiguous()

    def __repr__(self):
        return f"EvoNormS0 (num_features={self.num_features}, num_groups={self.num_groups}, eps={self.eps})"

class GroupNorm(nn.GroupNorm): #torch.nn.GroupNorm在导出onnx时只支持ndim>=3的tensor
    def forward(self,x):
        if x.ndim == 2:
            x = torch.unsqueeze(x,dim=-1)
            x = super().forward(x)
            x = torch.squeeze(x,dim=-1)
            return x
        else:
            return super().forward(x)

class EvoNormS01D(nn.Module):
    def __init__(self, num_groups,num_features, eps=1e-6, scale=True):
        super().__init__()
        self.num_groups = num_groups
        self.num_features = num_features
        if scale:
            self.gamma = nn.Parameter(torch.ones([1,num_groups,num_features//num_groups]))
        self.beta = nn.Parameter(torch.zeros([1,num_groups,num_features//num_groups]))
        self.v1 = nn.Parameter(torch.ones([1,num_groups,num_features//num_groups]))
        self.eps = eps
        self.scale = scale
    
    def forward(self, x):
        N,C = x.shape
        G = self.num_groups
        x = x.view([N,G,C//G])
        var = x.std(dim=(2),keepdim=True)
        gain = torch.rsqrt(var+self.eps)
        if self.scale:
            gain = gain*self.gamma
        
        x = x*torch.sigmoid(x*self.v1)*gain+self.beta

        return x.view([N,C]).contiguous()

    def __repr__(self):
        return f"EvoNormS01D (num_features={self.num_features}, num_groups={self.num_groups}, eps={self.eps})"

class SEBlock(nn.Module):
    def __init__(self,channels,r=16):
        super().__init__()
        self.channels = channels
        self.r = r
        self.fc0 = nn.Linear(self.channels,self.channels//r)
        self.fc1 = nn.Linear(self.channels//r,self.channels)

    def forward(self,net):
        org_net = net
        net = net.mean(dim=(2,3),keepdim=False)
        net = self.fc0(net)
        net = F.relu(net,inplace=True)
        net = self.fc1(net)
        net = F.sigmoid(net)
        net = torch.unsqueeze(net,dim=-1)
        net = torch.unsqueeze(net,dim=-1)
        return net*org_net

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256,max_rows=50,max_cols=50):
        super().__init__()
        self.row_embed = nn.Embedding(max_rows, num_pos_feats)
        self.col_embed = nn.Embedding(max_cols, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        '''

        Args:
            x: [...,C,H,W]
        Returns:

        '''
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)

        return pos

class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    It contains non-trainable buffers called
    "weight" and "bias", "running_mean", "running_var",
    initialized to perform identity transformation.

    The pre-trained backbone models from Caffe2 only contain "weight" and "bias",
    which are computed from the original four parameters of BN.
    The affine transform `x * weight + bias` will perform the equivalent
    computation of `(x - running_mean) / sqrt(running_var) * weight + bias`.
    When loading a backbone model from Caffe2, "running_mean" and "running_var"
    will be left unchanged as identity transformation.

    Other pre-trained backbone models may contain all 4 parameters.

    The forward is implemented by `F.batch_norm(..., training=False)`.
    """

    _version = 3

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        #self.register_buffer("weight", torch.ones(num_features))
        #self.register_buffer("bias", torch.zeros(num_features))
        self.weight = Parameter(torch.ones(num_features))
        self.bias = Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - eps)


    def forward(self, x):
        if x.requires_grad:
            # When gradients are needed, F.batch_norm will use extra memory
            # because its backward op computes gradients for weight/bias as well.
            scale = self.weight.float() * (self.running_var.float() + self.eps).rsqrt()
            bias = self.bias.float() - self.running_mean.float() * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            out_dtype = x.dtype  # may be half
            return x * scale.to(out_dtype) + bias.to(out_dtype)
        else:
            # When gradients are not needed, F.batch_norm is a single fused op
            # and provide more optimization opportunities.
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                eps=self.eps,
            )

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # No running_mean/var in early versions
            # This will silent the warnings
            if prefix + "running_mean" not in state_dict:
                state_dict[prefix + "running_mean"] = torch.zeros_like(self.running_mean)
            if prefix + "running_var" not in state_dict:
                state_dict[prefix + "running_var"] = torch.ones_like(self.running_var)

        # NOTE: if a checkpoint is trained with BatchNorm and loaded (together with
        # version number) to FrozenBatchNorm, running_var will be wrong. One solution
        # is to remove the version number from the checkpoint.
        if version is not None and version < 3:
            print("FrozenBatchNorm {} is upgraded to version 3.".format(prefix.rstrip(".")))
            # In version < 3, running_var are used without +eps.
            state_dict[prefix + "running_var"] -= self.eps

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def __repr__(self):
        return "FrozenBatchNorm2d(num_features={}, eps={})".format(self.num_features, self.eps)

    @classmethod
    def convert_frozen_batchnorm(cls, module):
        """
        Convert all BatchNorm/SyncBatchNorm in module into FrozenBatchNorm.

        Args:
            module (torch.nn.Module):

        Returns:
            If module is BatchNorm/SyncBatchNorm, returns a new module.
            Otherwise, in-place convert module and return it.

        Similar to convert_sync_batchnorm in
        https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
        """
        bn_module = nn.modules.batchnorm
        bn_module = (bn_module.BatchNorm2d, bn_module.SyncBatchNorm)
        res = module
        if isinstance(module, bn_module):
            res = cls(module.num_features)
            if module.affine:
                res.weight.data = module.weight.data.clone().detach()
                res.bias.data = module.bias.data.clone().detach()
            res.running_mean.data = module.running_mean.data
            res.running_var.data = module.running_var.data
            res.eps = module.eps
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozen_batchnorm(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res

class WSConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.view(weight.size(0),-1).mean(dim=1).view(-1,1,1,1)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    @classmethod
    def convert_wsconv(cls, module,exclude=None,parent=""):
        conv_module = nn.Conv2d
        res = module
        if isinstance(module, conv_module):
            res = cls(module.in_channels,module.out_channels,module.kernel_size,module.stride,
                      module.padding,module.dilation,module.groups,module.bias is not None)
            '''res.weight.data = module.weight.data
            if res.bias is not None:
                res.bias.data = module.bias.data'''
        else:
            for name, child in module.named_children():
                r_name = parent+"."+name if len(parent)>0 else name
                if exclude is not None:
                    if name in exclude or r_name in exclude:
                        print(f"Skip: {r_name}")
                        continue
                new_child = cls.convert_wsconv(child,exclude=exclude,parent=r_name)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res

class BCNorm(nn.Module):
    def __init__(self,num_features, num_groups=32,eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features=num_features,
                                 eps=eps,
                                 momentum=momentum,
                                 affine=affine,
                                 track_running_stats=track_running_stats)
        self.gn = nn.GroupNorm(num_groups=num_groups,num_channels=num_features,eps=eps,affine=affine)

    def forward(self,x):
        x = self.bn(x)
        return self.gn(x)

def get_norm(norm, out_channels,norm_args={}):
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.

    Returns:
        nn.Module or None: the normalization layer
    """
    if isinstance(norm,dict):
        norm = dict(norm)
        _norm = norm.pop('type')
        norm_args = norm
        norm = _norm
    if norm is None:
        return None
    if norm in ["GN","EvoNormS0"] and len(norm_args)==0:
        norm_args = {"num_groups":32}

    if norm == 'GN':
        #return nn.GroupNorm(num_channels=out_channels,**norm_args)
        if 'requires_grad' in norm_args:
            norm_args.pop('requires_grad')
        return GroupNorm(num_channels=out_channels,**norm_args)

    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": torch.nn.BatchNorm2d,
            # Fixed in https://github.com/pytorch/pytorch/pull/36382
            "SyncBN": nn.SyncBatchNorm,
            "FrozenBN": FrozenBatchNorm2d,
            # for debugging:
            "SyncBatchNorm": nn.SyncBatchNorm,
            "LayerNorm2d": LayerNorm2d,
            "LayerNorm":LayerNorm2d,
            "EvoNormS0": EvoNormS0,
            "InstanceNorm":nn.InstanceNorm2d,
        }[norm]
    return norm(num_features=out_channels,**norm_args)

def get_norm1d(norm, out_channels,norm_args={}):
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.

    Returns:
        nn.Module or None: the normalization layer
    """
    if norm is None:
        return None
    if norm in ["GN","EvoNormS0"] and len(norm_args)==0:
        norm_args = {"num_groups":32}

    if norm == 'GN':
        #return nn.GroupNorm(num_channels=out_channels,**norm_args)
        return GroupNorm(num_channels=out_channels,**norm_args)

    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": torch.nn.BatchNorm1d,
            "LayerNorm":nn.LayerNorm,
            "EvoNormS0": EvoNormS01D,
        }[norm]
    return norm(num_features=out_channels,**norm_args)

def get_conv_type(conv_cfg):
    if conv_cfg is None:
        return nn.Conv2d
    elif conv_cfg['type'] == "ConvWS":
        return ConvWS2d

class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="SiLU", inplace=True):
    if isinstance(name,dict):
        cfg = dict(name)
        name = cfg.pop('type')
        inplace = cfg.pop('inplace',True)
        assert len(cfg)==0,f"ERROR: activation cfg {cfg}"
    if name == "SiLU" or name == "Swish":
        module = nn.SiLU(inplace=inplace)
    elif name == "ReLU":
        module = nn.ReLU(inplace=inplace)
    elif name == "LeakyReLU":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif name == "Hardswish":
        module = nn.Hardswish(inplace=inplace)
    elif name == "GELU":
        module = nn.GELU()
    elif name == "HSigmoid":
        module = nn.Hardsigmoid(inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module

class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
        if not torch.jit.is_scripting():
            if x.numel() == 0 and self.training:
                # https://github.com/pytorch/pytorch/issues/12013
                assert not isinstance(
                    self.norm, torch.nn.SyncBatchNorm
                ), "SyncBatchNorm does not support empty inputs!"

        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, in_channels: int = 2048, num_heads: int=8, out_channels: int = 1024):
        '''
        in_channels: input_channels
        out_channels: output_channels
        spacial_dim: int/list, int: w=h=spacial_dim, list[int]: spacial_dim (h,w)
        '''
        super().__init__()
        if isinstance(spacial_dim,Iterable):
            spacial_size = spacial_dim[0]*spacial_dim[1] 
        else:
            spacial_size = spacial_dim ** 2
        self.positional_embedding = nn.Parameter(torch.randn(spacial_size + 1, in_channels) / in_channels ** 0.5)
        self.k_proj = nn.Linear(in_channels, in_channels)
        self.q_proj = nn.Linear(in_channels, in_channels)
        self.v_proj = nn.Linear(in_channels, in_channels)
        self.c_proj = nn.Linear(in_channels, out_channels or in_channels)
        self.num_heads = num_heads

    def forward(self, x,query=None):
        '''
        query: [B,C]
        return:
        [B,C]
        '''
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        if query is None:
            query = x.mean(dim=0, keepdim=True)
        else:
            query = torch.unsqueeze(query,dim=0)
        x = torch.cat([query, x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)

class NormalizedLinear(nn.Module):
    def __init__(self,in_channels,out_channels,eps=1e-5):
        super().__init__()
        self.weight = Parameter(torch.FloatTensor(out_channels,in_channels))
        nn.init.xavier_uniform_(self.weight)
        self.eps = eps
    
    @torch.cuda.amp.autocast(False)
    def forward(self,x):
        x = x.float()
        x = F.normalize(x,dim=-1)
        '''with torch.no_grad():
            temp_norm = torch.norm(
                self.weight, p=2,
                dim=1).unsqueeze(1).expand_as(self.weight)
            self.weight.div_(temp_norm + self.eps)
        return F.linear(x,self.weight)'''
        weight = F.normalize(self.weight,dim=1)
        return F.linear(x,weight)

    def normalize_weight(self):
        weight = F.normalize(self.weight,dim=1)
        return weight

    def loss(self):
        s = torch.eye(self.weight.shape[0])
        s = torch.ones_like(s)-s
        w = F.normalize(self.weight,dim=1)
        r = w@w.T
        l = r.clamp(min=0)
        l = l*s.to(l)
        return torch.mean(l)

class ArcMarginProductIF(nn.Module):
    '''
    insightface中ArcFace的实现
    '''
    def __init__(self, s=64.0, m=0.5):
        super().__init__()
        self.s = s
        self.m = m

    def forward(self, cosine: torch.Tensor, label):
        '''
        cosing: [M,...,C],[-1,1]
        label: [M,...],int,[0,C)
        '''
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cosine.acos_()
        cosine[index] += m_hot
        cosine.cos_().mul_(self.s)
        return cosine

class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            s: norm of input feature
            m: margin

            cos(theta + m)
        """
    def __init__(self, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.s = s
        self.m = m

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    @torch.cuda.amp.autocast(False)
    def forward(self, *,cosine, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        b = 1.0001 #防止cosine==1或者-1时梯度变为无穷大，无穷小
        max_v = 1.00004
        cosine = cosine.float()
        cosine = cosine.clamp(-max_v,max_v)
        sine = torch.sqrt((b - torch.pow(cosine, 2)).clamp(0, 1)).to(cosine.dtype)
        phi = cosine * self.cos_m - sine * self.sin_m #cos(theta+m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=cosine.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output

class ArcMarginProduct_(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            s: norm of input feature
            m: margin

            cos(theta + m)
        """
    def __init__(self, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.s = s
        self.m = m

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.id0 = Identity("id0")
        self.id1 = Identity("id1")
        self.id2 = Identity("id2")
        self.id3 = Identity("id3")
        self.id4 = Identity("id4")
        self.id5 = Identity("id5")

    @torch.cuda.amp.autocast(False)
    def forward(self, *,cosine, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        b = 1.00001 #防止cosine==1或者-1时梯度变为无穷大，无穷小
        cosine = cosine.float()
        cosine = self.id0(cosine)
        sine = torch.sqrt((b - torch.pow(cosine, 2)).clamp(0, 1)).to(cosine.dtype)
        sine = self.id1(sine)
        phi = cosine * self.cos_m - sine * self.sin_m #cos(theta+m)
        phi = self.id2(phi)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        phi = self.id3(phi)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=cosine.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output = self.id4(output)
        output *= self.s
        output = self.id5(output)

        return output

# Define the softmax_one function with added one in the denominator , which helps to reduce
#the negative impact impact of tiny values in the softmax function and improves numerical stability
def softmax_one(x, dim=None, _stacklevel=3, dtype=None):
    #subtract the max for stability
    x = x - x.max(dim=dim, keepdim=True).values
    #compute exponentials
    exp_x = torch.exp(x)
    #compute softmax values and add on in the denominator
    return exp_x / (1 + exp_x.sum(dim=dim, keepdim=True))

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class CHW2HWC(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        return x.permute(0,2,3,1)

class HWC2CHW(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        return x.permute(0,3,1,2)

class ParallelModule(nn.Module):
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], (OrderedDict,dict)):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def forward(self,x):
        res = []
        for k,m in self._modules.items():
            res.append(m(x))
        return res

class SumModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,xs):
        res = xs[0]
        for i in range(1,len(xs)):
            res += xs[i]
        return res

def hard_sigmoid(x):
        x = x/6+0.5
        x = torch.clamp(x,min=0,max=1)
        return x

class ChannelAttention(nn.Module):
    """Channel attention Module.

    Args:
        channels (int): The input (and output) channels of the attention layer.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None
    """

    def __init__(self, channels: int, init_cfg = None) -> None:
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function for ChannelAttention."""
        with torch.cuda.amp.autocast(enabled=False):
            out = self.global_avgpool(x)
        out = self.fc(out)
        out = self.act(out)
        '''if torch.jit.is_tracing:
            out = hard_sigmoid(out)
        else:
            out = self.act(out)'''
        return x * out


class MParent:
    def __init__(self,model):
        self.model = model

    def __getattr__(self, name):
        return self.model.__getattr__(name)

    def __getitem__(self, name):
        return self.model.__getitem__(name)

class Unsqueeze(nn.Module):
    def __init__(self,dim) -> None:
        super().__init__()
        self.dim = dim

    def forward(self,x):
        return torch.unsqueeze(x,dim=self.dim)

class Squeeze(nn.Module):
    def __init__(self,dim) -> None:
        super().__init__()
        self.dim = dim

    def forward(self,x):
        return torch.squeeze(x,dim=self.dim)

class Scale(nn.Module):
    """A learnable scale parameter.

    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.

    Args:
        scale (float): Initial value of scale factor. Default: 1.0
    """

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale