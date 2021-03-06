#coding=utf-8
from abc import ABCMeta, abstractmethod
import object_detection.fasterrcnn as fasterrcnn
import tensorflow as tf
import wml_tfutils as wmlt
import wnn
import object_detection.wlayers as odl
import object_detection.od_toolkit as od
import object_detection.utils as odu
from wtfop.wtfop_ops import wpad
import img_utils as wmli

slim = tf.contrib.slim

'''
Detectron2的默认实现中使用了与RCNN共享大部分特征提供参数，生成的mask 为14 x 14 (原文中使用的是不共享参数，输出28 x 28)
也就是说Mask分支使用的box为rpn输出的正样本
'''
class MaskRCNN(fasterrcnn.FasterRCNN):
    '''
    Detectron2实现中默认使用SIGMOID loss
    '''
    LT_SIGMOID=0
    LT_FOCAL_LOSS=1
    LT_USER_DEFINED=2
    def __init__(self,num_classes,input_shape,batch_size=1,loss_scale=1.0):
        super().__init__(num_classes,input_shape,batch_size)
        self.train_mask = False
        #[X,h,w]
        self.mask_logits = None
        #[X,h,w]
        self.finally_mask = None
        self.mask_loss_scale = loss_scale
        self.debug = True
        '''
        Mask分支Feature Map的输入
        Detectron2的实现中Mask分支与RCN分支共享提取特征部分，仅在最后一步有所不同
        RCN的Box分支只有一个avg pool + FC
        Mask分支只有一个ConvTranspose2d(channels=2048) + Conv2d
        '''
        self.mask_branch_input = None
        self.mask_loss_type = self.LT_SIGMOID
        '''
        user defined loss fn:input:[batch_size x h x w], [user_defined_loss_fn]
        output: [batch_size x h' x w']
        '''
        self.user_defined_mask_loss_fn = tf.nn.sigmoid_cross_entropy_with_logits
        print("mask loss scale:",self.mask_loss_scale)

    def train_mask_bn(self):
        return self.train_mask and self.train_bn
    '''
    注:这里实现与Detectron2中基本相同
    注:Detectron2中使用的原文Fig 4 left中的Head Architecture, 而原文使用的是Fig 4 Right的Head Architecture
    (Left: Mask与RCN共享大部分特征提取参数，仅最好输出部分不共享, Right不共享任何特征提取参数)
    这里的实现可以通过改变mask_branch_input的输入值来实现left/right head architecture的切换
    inputs:
    labels:[batch_size*box_nr], normal use None
    pmask:[batch_size*box_nr], normal use None
    size:[h,w], normal use None, _maskFeatureExtractor do the upsampling stuff
    net: replace the default input of mask branch
    outputs:
    mask_logits:[batch_size*pbox_nr,M,N]
    '''
    def buildMaskBranch(self,pmask=None,labels=None,size=None,reuse=False,net=None):
        if self.mask_branch_input is None:
            self.mask_branch_input = self.ssbp_net
        if net is None:
            net = self.mask_branch_input

        if self.train_mask and pmask is None and self.rcn_gtlabels is not None:
            pmask = tf.greater(self.rcn_gtlabels,0)
            pmask = tf.reshape(pmask,[-1])

        if labels is None:
            labels = tf.reshape(self.rcn_gtlabels,[-1])

        if pmask is not None:
            pmask = wmlt.assert_equal(pmask,[tf.shape(net)[:1],tf.shape(pmask)])
            net = tf.boolean_mask(net,pmask)
            labels = tf.boolean_mask(labels,pmask)

        if size is not None:
            net = tf.image.resize_bilinear(net,size)
        net = self._maskFeatureExtractor(net,reuse=reuse)
        net = tf.transpose(net,perm=(0,3,1,2))
        assert net.get_shape().as_list()[1]==self.num_classes-1,"Error dim size."
        self.mask_logits = wmlt.batch_gather(net,labels-1)
        return self.mask_logits
    '''
    labels:[batch_size*box_nr]
    pmask:[batch_size*box_nr]
    output:[batch_size*pbox_nr,M,N]
    this version, the mask branch and rcn network didn't share any wieghts.
    '''
    def buildMaskBranchV2(self,pmask=None,labels=None,bin_size=11,size=(33,33),reuse=False,roipooling=odl.DFROI()):
        base_net = self.base_net
        batch_index, batch_size, box_nr = self.rcn_batch_index_helper(self.proposal_boxes)
        net = roipooling(base_net, self.proposal_boxes, batch_index, bin_size, bin_size)
        net_channel = net.get_shape().as_list()[-1]
        net = wmlt.reshape(net, [batch_size * box_nr, bin_size, bin_size, net_channel])

        if self.train_mask and pmask is None:
            pmask = tf.greater(self.rcn_gtlabels,0)
            pmask = tf.reshape(pmask,[-1])

        if labels is None:
            labels = tf.reshape(self.rcn_gtlabels,[-1])

        if pmask is not None:
            pmask = wmlt.assert_equal(pmask,[tf.shape(net)[:1],tf.shape(pmask)])
            net = tf.boolean_mask(net,pmask)
            labels = tf.boolean_mask(labels,pmask)

        net = tf.image.resize_bilinear(net,size)
        net = self._maskFeatureExtractor(net,reuse=reuse)
        net = tf.transpose(net,perm=(0,3,1,2))
        assert net.get_shape().as_list()[1]==self.num_classes-1,"Error dim size."
        self.mask_logits = wmlt.batch_gather(net,labels-1)
        return self.mask_logits
    '''
    use the gtbboxes as the proposal box for mask branch (some third implement use this config)
    labels:[batch_size*box_nr]
    pmask:[batch_size*box_nr]
    net_process: process base net feature map (before rpn and rcn) 
    outputs:
    mask_logits:[batch_size*pbox_nr,M,N]
    gtlabels: selected gtlabels, [batch_size,pbox]
    gtboxes: selected gtboxes, [batch_size,pbox,4]
    indices: gtboxes's index in input gtbboxes, which means return gtboxes = batch_gather(input_gtboxes,indices)
    '''
    def buildMaskBranchV3(self,gtbboxes,gtlabels,gtlens,bboxes_nr,net=None,net_process=None,bin_size=11,size=(33,33),reuse=False,roipooling=odl.DFROI()):
        def random_select(labels,data_nr):
            with tf.name_scope("random_select"):
                if labels.dtype != tf.int32:
                    labels = tf.cast(labels,tf.int32)
                size = tf.shape(labels)[0]
                indexs = tf.range(data_nr)
                indexs = wpad(indexs, [0, size - data_nr])
                indexs = tf.random_shuffle(indexs)
                indexs = tf.random_crop(indexs, [bboxes_nr])
                labels = tf.gather(labels, indexs)
                return labels,indexs
        def batch_random_select(labels,data_nr):
            return tf.map_fn(lambda x:random_select(x[0],x[1]),elems=(labels,data_nr),dtype=(tf.int32,tf.int32),back_prop=False)
        if gtlens is not None:
            gtlabels,indices = batch_random_select(gtlabels,gtlens)
            gtbboxes = wmlt.batch_gather(gtbboxes,indices)
        else:
            indices = None
        if net is None:
            net = self.base_net
        batch_index, batch_size, box_nr = self.rcn_batch_index_helper(gtbboxes)
        net = roipooling(net, gtbboxes, batch_index, bin_size, bin_size)
        net_channel = net.get_shape().as_list()[-1]
        net = wmlt.reshape(net, [batch_size * box_nr, bin_size, bin_size, net_channel])
        if net_process is not None:
            net = net_process(net)
        net = tf.image.resize_bilinear(net,size)
        net = self._maskFeatureExtractor(net,reuse=reuse)
        net = tf.transpose(net,perm=(0,3,1,2))
        assert net.get_shape().as_list()[1]==self.num_classes-1,"Error dim size."
        gtlabels = tf.reshape(gtlabels,[-1])
        self.mask_logits = wmlt.batch_gather(net,gtlabels-1)
        return self.mask_logits,gtlabels,gtbboxes,indices

    '''
    output:[batch_size,X,H,W]
    '''
    def getBoxesAndMask(self,k=1000,mask_threshold=0.5,box_threshold=0.5,proposal_boxes=None,limits=None,
                   adjust_probability=None,nms=None,reuse=False,
                        size=(33,33)
                   ):
        self.getBoxesV2(k=k,
                        threshold=box_threshold,
                        proposal_boxes=proposal_boxes,
                        limits=limits,
                        adjust_probability=adjust_probability,
                        nms=nms)
        max_len = tf.maximum(1,tf.reduce_max(self.rcn_bboxes_lens))
        ssbp_net = wmlt.batch_gather(self.get_5d_mask_branch_net(),self.finally_indices[:,:max_len])
        ssbp_net = self.to_4d_mask_branch_net(ssbp_net)
        labels = self.finally_boxes_label[:,:max_len]
        labels = tf.reshape(labels,[-1])
        logits = self.buildMaskBranch(labels=labels,size=size,reuse=reuse,net=ssbp_net)
        mask = tf.greater(tf.sigmoid(logits),mask_threshold)
        mask = tf.cast(mask,tf.int32)
        shape = mask.get_shape().as_list()[1:]
        mask = wmlt.reshape(mask,[self.rcn_batch_size,max_len]+shape)
        self.finally_mask = mask
        return self.finally_mask

    def buildFakeMaskBranch(self,net=None):
        pmask = tf.ones(tf.shape(self.ssbp_net)[:1],dtype=tf.bool)
        labels = tf.ones(tf.shape(self.ssbp_net)[:1],dtype=tf.int32)
        self.buildMaskBranch(pmask,labels,size=[7,7],net=net)

    def buildFakeMaskBranchV2(self,net=None):
        pmask = tf.ones(tf.shape(self.ssbp_net)[:1],dtype=tf.bool)
        labels = tf.ones(tf.shape(self.ssbp_net)[:1],dtype=tf.int32)
        self.buildMaskBranchV2(pmask,labels,size=[7,7],net=net)


    '''
    net:[batch_size*box_nr,bin_size,bin_size,net_channel]
    output: [batch_size*box_nr,bin_size,bin_size,num_classes]
    '''
    @abstractmethod
    def _maskFeatureExtractor(self,net,reuse=False):
        with tf.variable_scope("MaskBranch",reuse=reuse):
            num_channels = net.get_shape().as_list()[-1]
            net = slim.convolution2d_transpose(net,num_channels,kernel_size=[2,2],
                                               stride=2,
                                               padding="SAME",
                                               normalizer_fn=None)
            net = slim.conv2d(net,self.num_classes,kernel_size=[3,3],stride=1,
                              activation_fn=None,
                              normalizer_fn=None)
            return net

    '''
    This implement is the same as Detectron2 
    '''
    def getMaskLoss(self,gtmasks,gtlabels=None):
        shape = self.mask_logits.get_shape().as_list()
        pmask = tf.greater(self.rcn_gtlabels,0)

        rcn_anchor_to_gt_indices = self.rcn_anchor_to_gt_indices
        rcn_anchor_to_gt_indices = tf.maximum(rcn_anchor_to_gt_indices,0)
        gtmasks = wmlt.batch_gather(gtmasks,rcn_anchor_to_gt_indices)
        bboxes = self.rcn_proposal_boxes
        gtmasks = tf.expand_dims(gtmasks,axis=-1)
        if self.debug:
            org_mask = tf.identity(gtmasks)
            org_mask = tf.reshape(org_mask,[-1]+org_mask.get_shape().as_list()[2:])
        gtmasks = wmlt.tf_crop_and_resize(gtmasks,bboxes,shape[1:3])
        gtmasks = tf.squeeze(gtmasks,axis=-1)

        gtmasks = tf.reshape(gtmasks,[-1]+gtmasks.get_shape().as_list()[2:])
        pmask = tf.reshape(pmask,[-1])
        if gtlabels is not None:
            gtlabels = wmlt.batch_gather(gtlabels,rcn_anchor_to_gt_indices)
            gtlabels = tf.cast(tf.reshape(gtlabels,[-1]),tf.int32)
            cgtlabels = tf.cast(tf.reshape(self.rcn_gtlabels,[-1]),tf.int32)
            gtmasks = wmlt.assert_equal(gtmasks,[tf.boolean_mask(gtlabels,pmask),tf.boolean_mask(cgtlabels,pmask)],"ASSERT_GTLABELS_EQUAL")
        gtmasks = tf.boolean_mask(gtmasks,pmask)
        log_mask = tf.expand_dims(gtmasks,axis=-1)
        if self.debug:
            log_boxes = tf.expand_dims(tf.reshape(bboxes,[-1,4]),axis=1)
            log_mask1 = odu.tf_draw_image_with_box(org_mask,log_boxes,scale=False)
            log_mask1 = tf.boolean_mask(log_mask1,pmask)
            log_mask = wmli.concat_images([log_mask1,log_mask])
        wmlt.image_summaries(log_mask,"mask",max_outputs=5)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=gtmasks,logits=self.mask_logits)
        loss = tf.reduce_mean(loss)
        tf.losses.add_loss(loss*self.loss_scale)
        return loss

    '''
    gtmasks:[batch_size,img_h,img_w,c] is the same size as the input images
    '''
    def getMaskLossV2(self,gtbboxes,gtmasks,indices,scope="Loss"):
        with tf.variable_scope(scope,default_name="MaskLoss"):
            shape = self.mask_logits.get_shape().as_list()

            if indices is not None:
                gtmasks = wmlt.batch_gather(gtmasks,indices)
            gtmasks = tf.expand_dims(gtmasks,axis=-1)
            if self.debug:
                org_mask = tf.identity(gtmasks)
                org_mask = tf.reshape(org_mask,[-1]+org_mask.get_shape().as_list()[2:])
            gtmasks = wmlt.tf_crop_and_resize(gtmasks,gtbboxes,shape[1:3])
            gtmasks = tf.squeeze(gtmasks,axis=-1)

            gtmasks = tf.reshape(gtmasks,[-1]+gtmasks.get_shape().as_list()[2:])
            log_mask  = tf.expand_dims(gtmasks,axis=-1)
            if self.debug:
                log_boxes = tf.expand_dims(tf.reshape(gtbboxes,[-1,4]),axis=1)
                log_mask1 = odu.tf_draw_image_with_box(org_mask,log_boxes,scale=False)
                log_mask = wmli.concat_images([log_mask1,log_mask])
            wmlt.image_summaries(log_mask,"mask",max_outputs=5)
            if self.mask_loss_type == self.LT_SIGMOID:
                print("Use sigmoid mask loss.")
                loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=gtmasks,logits=self.mask_logits)
            elif self.mask_loss_type == self.LT_FOCAL_LOSS:
                print("Use focal mask loss")
                loss = wnn.sigmoid_cross_entropy_with_logits_FL(labels=gtmasks,logits=self.mask_logits)
            elif self.mask_loss_type == self.LT_USER_DEFINED:
                print("Use user defined mask loss")
                loss = self.user_defined_mask_loss_fn(labels=gtmasks,logits=self.mask_logits)
            loss = tf.reduce_sum(loss,axis=[1,2])
            loss = tf.reduce_mean(loss)*self.mask_loss_scale
            tf.losses.add_loss(loss)
            return loss

    def getRCNBoxes(self):
        probs = tf.nn.softmax(self.rcn_logits)
        boxes,_,_  = od.get_predictionv4(class_prediction=probs,bboxes_regs=self.rcn_regs,
                                        proposal_bboxes=self.proposal_boxes,classes_wise=self.pred_bboxes_classwise)
        return tf.stop_gradient(boxes)

    def getLoss(self,gtmasks,gtlabels=None,use_scores=False):
        pc_loss, pr_loss, nc_loss, psize_div_all = self.getRCNLoss(use_scores=use_scores)
        mask_loss = self.getMaskLoss(gtmasks=gtmasks,gtlabels=gtlabels)
        return mask_loss,pc_loss, pr_loss, nc_loss, psize_div_all

    '''
    use buildMaskBranchV3/buildFakeMaskBranch for mask branch
    in this version, the input gtboxes and indices is the return value of buildMaskBranchV3
    the gtmasks is the ground truth data.
    '''
    def getLossV2(self,gtbboxes,gtmasks,indices,scores=None,od_loss=None,scope="Loss"):
        pc_loss, pr_loss, nc_loss, psize_div_all = self.getRCNLoss(scores=scores,scope=scope,loss=od_loss)
        mask_loss = self.getMaskLossV2(gtbboxes=gtbboxes,gtmasks=gtmasks,indices=indices,scope=scope)
        return mask_loss,pc_loss, pr_loss, nc_loss, psize_div_all

    def get_5d_mask_branch_net(self):
        if self.mask_branch_input is None:
            self.mask_branch_input = self.ssbp_net
        shape = self.mask_branch_input.get_shape().as_list()[1:]
        return wmlt.reshape(self.mask_branch_input,[self.rcn_batch_size,self.rcn_box_nr]+shape)

    def to_4d_mask_branch_net(self,net):
        shape = net.get_shape().as_list()[2:]
        return wmlt.reshape(net,[-1]+shape)
