import math
import tensorflow as tf
import wmodule
import functools
from .backbone import Backbone
from .build import BACKBONE_REGISTRY,build_backbone_hook,build_backbone_by_name
from .resnet import build_resnet_backbone
from .shufflenetv2 import build_shufflenetv2_backbone
from .efficientnet import build_efficientnet_backbone
import collections
import object_detection2.od_toolkit as odt
import object_detection2.architectures_tools as odat

slim = tf.contrib.slim

class BIFPN(Backbone):
    """
    This module implements Feature Pyramid Network.
    It creates pyramid features built on top of some input feature maps.
    """

    def __init__(
        self, cfg,bottom_up, in_features, out_channels,
            parent=None,*args,**kwargs
    ):
        """
        Args:
            bottom_up (Backbone): module representing the bottom up subnetwork.
                Must be a subclass of :class:`Backbone`. The multi-scale feature
                maps generated by the bottom up network, and listed in `in_features`,
                are used to generate BIFPN levels.
            in_features (list[str]): names of the input feature maps coming
                from the backbone to which BIFPN is attached. For example, if the
                backbone produces ["res2", "res3", "res4"], any *contiguous* sublist
                of these may be used; order must be from high to low resolution.
            out_channels (int): number of channels in the output feature maps.
        """
        stage = int(in_features[-1][1:])
        super(BIFPN, self).__init__(cfg,parent=parent,*args,**kwargs)
        assert isinstance(bottom_up, Backbone)

        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.in_features = in_features
        self.bottom_up = bottom_up
        self.out_channels = out_channels
        self.scope = "BIFPN"
        self.use_depthwise = False
        self.interpolate_op=tf.image.resize_nearest_neighbor
        self.stage = stage
        self.normalizer_fn,self.norm_params = odt.get_norm(self.cfg.MODEL.BIFPN.NORM,self.is_training)
        self.hook_before,self.hook_after = build_backbone_hook(cfg.MODEL.BIFPN,parent=self)
        self.activation_fn = odt.get_activation_fn(self.cfg.MODEL.BIFPN.ACTIVATION_FN)

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, x):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to BIFPN feature map tensor
                in high to low resolution order. Returned feature names follow the BIFPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        # Reverse feature maps into top-down order (from low to high resolution)
        bottom_up_features = self.bottom_up(x)
        if self.hook_before is not None:
            bottom_up_features = self.hook_before(bottom_up_features,x)
        image_features = [bottom_up_features[f] for f in self.in_features]
        use_depthwise = self.use_depthwise
        depth = self.out_channels
        feature_maps = []

        if self.normalizer_fn is not None:
            normalizer_fn = functools.partial(self.normalizer_fn,**self.norm_params)
        else:
            normalizer_fn = None

        padding = 'SAME'
        weight_decay = 1e-4

        if use_depthwise:
            conv_op = functools.partial(slim.separable_conv2d, depth_multiplier=1,
                                        normalizer_fn=normalizer_fn,
                                        padding=padding)
        else:
            conv_op = functools.partial(slim.conv2d,
                                        weights_regularizer=slim.l2_regularizer(weight_decay),
                                        normalizer_fn=normalizer_fn,
                                        padding=padding)

        with tf.variable_scope(self.scope,"BIFPN"):
            for i,net in enumerate(image_features):
                '''
                FPN的官方实现没有使用norm与激活函数，但EfficientDet的官方实现中使用了BN,没有使用激活函数
                kernel都为 1 x 1
                '''
                net = slim.conv2d(net,depth,
                                  kernel_size=1,
                                  normalizer_fn=normalizer_fn,
                                  activation_fn=None,
                                  scope=f"projection_{i}")
                feature_maps.append(net)
            feature_maps.reverse()
            '''
            原文写的D0重复两次，D3重复5次，但实际相应的实现D0重复了3次，D3重复了6次，这里按作者的源码保持一致
            '''
            #repeat = self.cfg.MODEL.BIFPN.REPEAT+1
            repeat = self.cfg.MODEL.BIFPN.REPEAT

            for i in range(repeat):
                feature_maps = odat.BiFPN(feature_maps,
                                          conv_op=conv_op,
                                          activation_fn=self.activation_fn,
                                          scope=f"cell_{i}")
            feature_maps.reverse()

            res = collections.OrderedDict()
            for name,net in zip(self.in_features,feature_maps):
                index = int(name[1:])
                res[f"P{index}"] = net
            if self.hook_after is not None:
                res = self.hook_after(res, x)
            return res


@BACKBONE_REGISTRY.register()
def build_efficientnet_bifpn_backbone(cfg,*args,**kwargs):
    """
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_efficientnet_backbone(cfg,*args,**kwargs)
    in_features = cfg.MODEL.BIFPN.IN_FEATURES
    out_channels = cfg.MODEL.BIFPN.OUT_CHANNELS
    backbone = BIFPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        cfg=cfg,
        *args,
        **kwargs
    )
    return backbone

@BACKBONE_REGISTRY.register()
def build_resnet_bifpn_backbone(cfg,*args,**kwargs):
    """
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resnet_backbone(cfg,*args,**kwargs)
    in_features = cfg.MODEL.BIFPN.IN_FEATURES
    out_channels = cfg.MODEL.BIFPN.OUT_CHANNELS
    backbone = BIFPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        cfg=cfg,
        *args,
        **kwargs
    )
    return backbone

@BACKBONE_REGISTRY.register()
def build_shufflenetv2_bifpn_backbone(cfg,*args,**kwargs):
    """
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_shufflenetv2_backbone(cfg,*args,**kwargs)
    in_features = cfg.MODEL.BIFPN.IN_FEATURES
    out_channels = cfg.MODEL.BIFPN.OUT_CHANNELS
    backbone = BIFPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        cfg=cfg,
        *args,
        **kwargs
    )
    return backbone

@BACKBONE_REGISTRY.register()
def build_any_bifpn_backbone(cfg,*args,**kwargs):
    """
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_backbone_by_name(cfg.MODEL.BIFPN.BACKBONE,cfg, *args,**kwargs)
    in_features = cfg.MODEL.BIFPN.IN_FEATURES
    out_channels = cfg.MODEL.BIFPN.OUT_CHANNELS
    backbone = BIFPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        cfg=cfg,
        *args,
        **kwargs
    )
    return backbone
