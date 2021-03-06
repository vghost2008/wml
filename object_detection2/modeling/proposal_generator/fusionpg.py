#coding=utf-8
import tensorflow as tf
import wmodule
from .build import PROPOSAL_GENERATOR_REGISTRY,build_proposal_generator_by_name
from object_detection2.modeling.proposal_generator.rpn_outputs import find_top_rpn_proposals
from object_detection2.modeling.onestage_heads.retinanet_outputs import *
from object_detection2.datadef import *
from basic_tftools import batch_size

slim = tf.contrib.slim

__all__ = ["FusionPG"]

@PROPOSAL_GENERATOR_REGISTRY.register()
class FusionPG(wmodule.WChildModule):
    """
    Implement FCSOPG: Fully Convolutional One-Stage Object Detection
    """

    def __init__(self, cfg,parent,*args,**kwargs):
        super().__init__(cfg,parent,*args,**kwargs)
        self.pgs = [build_proposal_generator_by_name(name,cfg,parent=self,*args,**kwargs) for name in cfg.MODEL.FUSIONPG.NAMES]
        self.pre_nms_topk = {
            True: self.cfg.MODEL.FUSIONPG.PRE_NMS_TOPK_TRAIN,
            False: self.cfg.MODEL.FUSIONPG.PRE_NMS_TOPK_TEST,
        }
        self.post_nms_topk = {
            True: self.cfg.MODEL.FUSIONPG.POST_NMS_TOPK_TRAIN,
            False: self.cfg.MODEL.FUSIONPG.POST_NMS_TOPK_TEST,
        }
        self.nms_thresh = self.cfg.MODEL.FUSIONPG.NMS_THRESH_TEST

    def forward(self, batched_inputs,features):
        boxes = []
        losses = {}
        probabilitys = []
        B = batch_size(batched_inputs[IMAGE])
        bboxes_nrs = []
        for i,pg in enumerate(self.pgs):
            result_i,losses_i = pg(batched_inputs,features)
            boxes_i = result_i[PD_BOXES]
            wsummary.histogram_or_scalar(result_i[PD_PROBABILITY],self.cfg.MODEL.FUSIONPG.NAMES[i]+"_scores")
            boxes.append(boxes_i)
            for k,v in losses_i.items():
                losses[self.cfg.MODEL.FUSIONPG.NAMES[i]+"_"+k] = v
            bboxes_nrs.append(tf.shape(boxes_i)[1])

        bboxes_nr = tf.to_float(tf.reduce_max(tf.convert_to_tensor(bboxes_nrs)))+1e-8
        for i, pg in enumerate(self.pgs):
            probabilitys_i = tf.range(bboxes_nrs[i])
            probabilitys_i = 1.0-tf.to_float(probabilitys_i)/bboxes_nr
            probabilitys_i = tf.expand_dims(probabilitys_i,axis=0)
            probabilitys_i = tf.tile(probabilitys_i,[B,1])
            probabilitys.append(probabilitys_i)

        proposals, logits = find_top_rpn_proposals(
            tf.concat(boxes,axis=1),
            tf.concat(probabilitys,axis=1),
            self.nms_thresh,
            self.pre_nms_topk[self.is_training],
            self.post_nms_topk[self.is_training],
            None
        )
        outdata = {PD_BOXES: proposals, PD_PROBABILITY: tf.nn.sigmoid(logits)}
        wsummary.detection_image_summary(images=batched_inputs[IMAGE],
                                         boxes=outdata[PD_BOXES],
                                         name="fusionpg/proposals")
        wsummary.detection_image_summary(images=batched_inputs[IMAGE],
                                         boxes=outdata[PD_BOXES][:,:20],
                                         name="fusionpg/proposals")

        return outdata, losses
