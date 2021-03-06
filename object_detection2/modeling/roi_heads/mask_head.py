#coding=utf-8
from thirdparty.registry import Registry
import wmodule
import tensorflow as tf
import wml_tfutils as wmlt
from object_detection2.datadef import *
import object_detection2.config.config as config
import image_visualization as ivis
import wsummary
import img_utils as wmli
import object_detection2.od_toolkit as odt
import basic_tftools as btf
from basic_tftools import channel as get_channel

slim = tf.contrib.slim

ROI_MASK_HEAD_REGISTRY = Registry("ROI_MASK_HEAD")
ROI_MASK_HEAD_REGISTRY.__doc__ = """
Registry for mask heads, which predicts instance masks given
per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""


@wmlt.add_name_scope
def mask_rcnn_loss(inputs,pred_mask_logits, proposals:EncodedData,fg_selection_mask,log=True):
    '''

    :param inputs:inputs[GT_MASKS] [batch_size,N,H,W]
    :param pred_mask_logits: [Y,H,W,C] C==1 if cls_anostic_mask else num_classes, H,W is the size of mask
       not the position in org image
    :param proposals:proposals.indices:[batch_size,M], proposals.boxes [batch_size,M],proposals.gt_object_logits:[batch_size,M]
    :param fg_selection_mask: [X]
    X = batch_size*M
    Y = tf.reduce_sum(fg_selection_mask)
    :return:
    '''
    cls_agnostic_mask = pred_mask_logits.get_shape().as_list()[-1] == 1
    total_num_masks,mask_H,mask_W,C  = wmlt.combined_static_and_dynamic_shape(pred_mask_logits)
    assert mask_H==mask_W, "Mask prediction must be square!"

    gt_masks = inputs[GT_MASKS] #[batch_size,N,H,W]

    with tf.device("/cpu:0"):
        #当输入图像分辨率很高时这里可能会消耗过多的GPU资源，因此改在CPU上执行
        batch_size,X,H,W = wmlt.combined_static_and_dynamic_shape(gt_masks)
        #background include in proposals, which's indices is -1
        gt_masks = tf.reshape(gt_masks,[batch_size*X,H,W])
        indices = btf.twod_indexs_to_oned_indexs(tf.nn.relu(proposals.indices),depth=X)
        indices = tf.boolean_mask(indices,fg_selection_mask)
        gt_masks = tf.gather(gt_masks,indices)

    boxes = proposals.boxes
    batch_size,box_nr,box_dim = wmlt.combined_static_and_dynamic_shape(boxes)
    boxes = tf.reshape(boxes,[batch_size*box_nr,box_dim])
    boxes = tf.boolean_mask(boxes,fg_selection_mask)

    with tf.device("/cpu:0"):
        #当输入图像分辨率很高时这里可能会消耗过多的GPU资源，因此改在CPU上执行
        gt_masks = tf.expand_dims(gt_masks,axis=-1)
        croped_masks_gt_masks = wmlt.tf_crop_and_resize(gt_masks,boxes,[mask_H,mask_W])

    if not cls_agnostic_mask:
        gt_classes = proposals.gt_object_logits
        gt_classes = tf.reshape(gt_classes,[-1])
        gt_classes = tf.boolean_mask(gt_classes,fg_selection_mask)
        pred_mask_logits = tf.transpose(pred_mask_logits,[0,3,1,2])
        pred_mask_logits = wmlt.batch_gather(pred_mask_logits,gt_classes-1) #预测中不包含背景
        pred_mask_logits = tf.expand_dims(pred_mask_logits,axis=-1)


    if log and config.global_cfg.GLOBAL.SUMMARY_LEVEL<=SummaryLevel.DEBUG:
        with tf.device(":/cpu:0"):
            with tf.name_scope("mask_loss_summary"):
                pmasks_2d = tf.reshape(fg_selection_mask,[batch_size,box_nr])
                boxes_3d = tf.expand_dims(boxes,axis=1)
                wsummary.positive_box_on_images_summary(inputs[IMAGE],proposals.boxes,
                                                        pmasks=pmasks_2d)
                image = wmlt.select_image_by_mask(inputs[IMAGE],pmasks_2d)
                t_gt_masks = tf.expand_dims(tf.squeeze(gt_masks,axis=-1),axis=1)
                wsummary.detection_image_summary(images=image,boxes=boxes_3d,instance_masks=t_gt_masks,
                                                 name="mask_and_boxes_in_mask_loss")
                log_mask = gt_masks
                log_mask = ivis.draw_detection_image_summary(log_mask,boxes=tf.expand_dims(boxes,axis=1))
                log_mask = wmli.concat_images([log_mask, croped_masks_gt_masks])
                wmlt.image_summaries(log_mask,"mask",max_outputs=3)

                log_mask = wmli.concat_images([gt_masks, tf.cast(pred_mask_logits>0.5,tf.float32)])
                wmlt.image_summaries(log_mask,"gt_vs_pred",max_outputs=3)
    mask_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=croped_masks_gt_masks,logits=pred_mask_logits)
    mask_loss = btf.safe_reduce_mean(mask_loss)

    return mask_loss
    pass

@wmlt.add_name_scope
def mask_rcnn_inference(pred_mask_logits, pred_instances):
    """
    Convert pred_mask_logits to estimated foreground probability masks while also
    extracting only the masks for the predicted classes in pred_instances. For each
    predicted box, the mask of the same class is attached to the instance by adding a
    new RD_MASKS field to pred_instances.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B,Hmask, Wmask,C) or (B, Hmask, Wmask, 1)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        pred_instances (dict): A dict of prediction results, pred_instances[RD_LABELS]:[batch_size,Y],
        pred_instances[RD_LENGTH], [batch_size]
        current the batch_size must be 1, and X == pred_instances[RD_LENGTH][0] == Y
    Returns:
        None. pred_instances will contain an extra RD_MASKS field storing a mask of size [batch_size,Y,Hmask,
            Wmask] for predicted class. Note that the masks are returned as a soft (non-quantized)
            masks the resolution predicted by the network; post-processing steps, such as resizing
            the predicted masks to the original image resolution and/or binarizing them, is left
            to the caller.
    """
    cls_agnostic_mask = pred_mask_logits.get_shape().as_list()[-1] == 1
    labels = pred_instances[RD_LABELS]
    batch_size,box_nr = wmlt.combined_static_and_dynamic_shape(labels)
    if not cls_agnostic_mask:
        # Select masks corresponding to the predicted classes
        pred_mask_logits = tf.transpose(pred_mask_logits,[0,3,1,2])
        labels = tf.reshape(labels,[-1])-1 #去掉背景
        #当同时预测多个图片时，labels后面可能有填充的0，上一步减1时可能出现负数
        pred_mask_logits = wmlt.batch_gather(pred_mask_logits,tf.nn.relu(labels))
    total_box_nr,H,W = wmlt.combined_static_and_dynamic_shape(pred_mask_logits)
    pred_mask_logits = tf.reshape(pred_mask_logits,[batch_size,box_nr,H,W])

    pred_mask_logits = tf.nn.sigmoid(pred_mask_logits)

    pred_instances[RD_MASKS] = pred_mask_logits


@ROI_MASK_HEAD_REGISTRY.register()
class MaskRCNNConvUpsampleHead(wmodule.WChildModule):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    """

    def __init__(self, cfg,**kwargs):
        """
        The following attributes are parsed from config:
            num_conv: the number of conv layers
            conv_dim: the dimension of the conv layers
            norm: normalization for the conv layers
        """
        super(MaskRCNNConvUpsampleHead, self).__init__(cfg,**kwargs)
        #Detectron2默认没有使用normalizer, 使用测试数据发现是否使用normalizer并没有什么影响
        self.normalizer_fn,self.norm_params = odt.get_norm(self.cfg.MODEL.ROI_MASK_HEAD.NORM,self.is_training)
        self.activation_fn = odt.get_activation_fn(self.cfg.MODEL.ROI_MASK_HEAD.ACTIVATION_FN)


    def forward(self, x):
        cfg = self.cfg
        # fmt: off
        num_classes       = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        conv_dims         = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        num_conv          = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        cls_agnostic_mask = cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK
        # fmt: on
        num_mask_classes = 1 if cls_agnostic_mask else num_classes

        with tf.variable_scope("MaskHead"):
            for k in range(num_conv):
                x = slim.conv2d(x,conv_dims,[3,3],padding="SAME",
                                    activation_fn=self.activation_fn,
                                    normalizer_fn=self.normalizer_fn,
                                    normalizer_params=self.norm_params,
                                    scope=f"Conv{k}")
            x = slim.conv2d_transpose(x,conv_dims,kernel_size=2,
                                    stride=2,
                                    activation_fn=self.activation_fn,
                                    normalizer_fn=self.normalizer_fn,
                                    normalizer_params=self.norm_params,
                                    scope="Upsample")
            x = slim.conv2d(x,num_mask_classes,kernel_size=1,activation_fn=None,normalizer_fn=None,
                            scope="predictor")
        return x

@ROI_MASK_HEAD_REGISTRY.register()
class HighResolutionMaskHead(wmodule.WChildModule):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    """

    def __init__(self, cfg,**kwargs):
        """
        The following attributes are parsed from config:
            num_conv: the number of conv layers
            conv_dim: the dimension of the conv layers
            norm: normalization for the conv layers
        """
        super(HighResolutionMaskHead, self).__init__(cfg,**kwargs)
        self.normalizer_fn,self.norm_params = odt.get_norm(self.cfg.MODEL.ROI_MASK_HEAD.NORM,self.is_training)
        self.activation_fn = odt.get_activation_fn(self.cfg.MODEL.ROI_MASK_HEAD.ACTIVATION_FN)


    def forward(self, x):
        cfg = self.cfg
        # fmt: off
        num_classes       = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        conv_dims         = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        num_conv          = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        cls_agnostic_mask = cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK
        # fmt: on
        num_mask_classes = 1 if cls_agnostic_mask else num_classes
        x_channel = get_channel(x)
        assert x_channel>conv_dims,"error conv dims for mask."

        with tf.variable_scope("MaskHead"):
            x_identity,x = tf.split(x,num_or_size_splits=[x_channel-conv_dims,conv_dims],axis=-1)
            x_iden_channel = get_channel(x_identity)
            if 'G' in self.norm_params and self.norm_params['G'] > get_channel(x):
                self.norm_params['G'] = get_channel(x)
            for k in range(num_conv):
                x = slim.conv2d(x,conv_dims,[3,3],padding="SAME",
                                activation_fn=self.activation_fn,
                                normalizer_fn=self.normalizer_fn,
                                normalizer_params=self.norm_params,
                                scope=f"Conv{k}")
            x = slim.conv2d_transpose(x,x_iden_channel,kernel_size=2,
                                      stride=2,
                                      activation_fn=self.activation_fn,
                                      normalizer_fn=self.normalizer_fn,
                                      normalizer_params=self.norm_params,
                                      scope="Upsample")
            B,H,W,C = btf.combined_static_and_dynamic_shape(x)
            x_identity = tf.image.resize_bilinear(x_identity,size=(H,W))
            x = x_identity+x
            x = slim.conv2d(x,num_mask_classes,kernel_size=1,activation_fn=None,normalizer_fn=None,
                            scope="predictor")
        return x

def build_mask_head(cfg,*args,**kwargs):
    """
    Build a mask head defined by `cfg.MODEL.ROI_MASK_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_MASK_HEAD.NAME
    return ROI_MASK_HEAD_REGISTRY.get(name)(cfg,*args,**kwargs)
