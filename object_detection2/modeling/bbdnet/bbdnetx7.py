import numpy as np
import tensorflow as tf
from object_detection2.odtools import *
from GN.graph import DynamicAdjacentMatrix,DynamicAdjacentMatrixShallow
from object_detection2.modeling.roi_heads.build import ROI_BOX_HEAD_REGISTRY
from object_detection2.modeling.poolers import ROIPooler
import wnn
import image_visualization as imv
import wsummary
import object_detection.bboxes as odb
from wtfop.wtfop_ops import adjacent_matrix_generator_by_iouv3
import wnnlayer as wnnl
import wml_tfutils as wmlt
import basic_tftools as btf
from functools import partial
from .abstractbbdnetx7 import AbstractBBDNet
import nlp.wlayers as nlpl
from wmodule import WModule
from object_detection2.standard_names import *
import object_detection2.od_toolkit as odt
import wtfop.wtfop_ops as wop
import object_detection2.wlayers as odl
import functools
import math
from object_detection2.data.dataloader import DataLoader
from .build import BBDNET_MODEL
import nlp.wlayers as nl


slim = tf.contrib.slim


'''
Each time only process one example.
'''
class BBDNetForOneImg(AbstractBBDNet):
    '''
    boxes: the  boxes, [batch_size=1,k,4]
    probability: [batch_size=1,k,classes_num] the probability of boxes
    map_data:[batch_size=1,k,C]
    classes_num: ...
    '''

    def __init__(self, cfg,boxes, map_data):
        super().__init__(cfg,boxes, map_data)
        self.mid_edges_outputs = []
        self.mid_global_outputs = []
        self.mid_nms_outputs= []
        self.normalizer_fn,self.normalizer_params = wnnl.graph_norm,{}
        self.activation_fn = tf.nn.leaky_relu
        self.weight_decay = 1e-4
        self.POINT_HIDDEN_SIZE,self.EDGE_HIDDEN_SIZE,self.GLOBAL_HIDDEN_SIZE = self.cfg.MODEL.BBDNET.DIMS

        with tf.variable_scope("BBDNet",reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.fully_connected],
                                normalizer_fn=None,
                                normalizer_params=None,
                                activation_fn=self.activation_fn):
                self.build_net()

    def build_net(self):
        adj_mt = adjacent_matrix_generator_by_iouv3(bboxes=self.boxes,
                                                    threshold=0.3,
                                                    keep_connect=False)
        self.adj_mt = adj_mt
        if len(self.map_data.get_shape()) == 4:
            Nr,H,W,C = wmlt.combined_static_and_dynamic_shape(self.map_data)
            map_data = tf.reshape(self.map_data,[Nr,H*W*C])
            self.map_data = slim.fully_connected(map_data,self.POINT_HIDDEN_SIZE)

        with tf.variable_scope("NodeEncode"):
            # node encode
            cxywh_boxes = odb.to_cxyhw(self.boxes)
            pos_data = tf.concat([self.boxes, cxywh_boxes], axis=1)
            net0 = slim.fully_connected(pos_data, 128,weights_regularizer=slim.l2_regularizer(self.weight_decay))

            # Fusion all parts of node.
            net = tf.concat([net0, self.map_data], axis=1,name="concat_net0_net2")
            #net = tf.zeros_like(net)
            points_data = self._mlp(net, dims=self.POINT_HIDDEN_SIZE, scope="MLP_a")

        unit_nr = self.cfg.MODEL.BBDNET.RES_UNIT_NR
        edge_fn = partial(self.res_block, dims=self.EDGE_HIDDEN_SIZE, unit_nr=unit_nr, scope="UpdateEdge")
        point_fn = partial(self.res_block, dims=self.POINT_HIDDEN_SIZE, unit_nr=unit_nr, scope="UpdatePoint")

        grapy_t = DynamicAdjacentMatrix

        self.A = grapy_t(adj_mt=adj_mt,
                         points_data=points_data,
                         edges_data=None,
                         edges_data_dim=self.EDGE_HIDDEN_SIZE)
        self.A.use_sent_edges_for_node = self.cfg.MODEL.BBDNET.USE_SENT_EDGES_FOR_NODE
        self.A.redges_reducer_for_points = functools.partial(tf.reduce_sum,axis=0,keepdims=False)
        self.A.sedges_reducer_for_points = functools.partial(tf.reduce_sum,axis=0,keepdims=False)

        if self.cfg.MODEL.BBDNET.EDGES_REDUCER_FOR_POINTS == "sum":
            print("sum reducer")
            self.A.edges_reducer_for_points = tf.unsorted_segment_sum
        elif self.cfg.MODEL.BBDNET.EDGES_REDUCER_FOR_POINTS == "mean":
            print("mean reducer")
            self.A.edges_reducer_for_points = tf.unsorted_segment_mean

        self.A.global_attr = None

        # edge encode
        with tf.variable_scope("EdgeEncode"):
            senders_indexs, receivers_indexs = self.A.senders_indexs, self.A.receivers_indexs
            senders_bboxes = tf.gather(self.boxes, senders_indexs)
            receivers_bboxes = tf.gather(self.boxes, receivers_indexs)
            points_data0 = tf.concat([self.boxes, cxywh_boxes[:, :2]], axis=1)
            points_data1 = self.map_data
            iou = odb.batch_bboxes_jaccard(senders_bboxes, receivers_bboxes)
            iou = tf.expand_dims(iou, axis=-1)
            iou = slim.fully_connected(iou,self.EDGE_HIDDEN_SIZE // 2,scope="encode_iou",
                                           weights_regularizer=slim.l2_regularizer(self.weight_decay))

            e_data0_s = tf.gather(points_data0, senders_indexs)
            e_data0_r = tf.gather(points_data0, receivers_indexs)
            e_data0 = e_data0_r - e_data0_s
            e_data0 = slim.fully_connected(e_data0,self.EDGE_HIDDEN_SIZE // 2,scope="MLP_a",
                                           weights_regularizer=slim.l2_regularizer(self.weight_decay))
            e_data1_s = tf.gather(points_data1, senders_indexs)
            e_data1_r = tf.gather(points_data1, receivers_indexs)
            e_data1 = (e_data1_r + e_data1_s) / 2.0
            e_data1 = slim.fully_connected(e_data1,self.EDGE_HIDDEN_SIZE // 2,scope="MLP_b",
                                           weights_regularizer=slim.l2_regularizer(self.weight_decay))
            e_data = tf.concat([e_data0, e_data1, iou], axis=1)
            e_data = slim.fully_connected(e_data, self.EDGE_HIDDEN_SIZE, scope="EdgeEncode",
                                          weights_regularizer=slim.l2_regularizer(self.weight_decay))
            self.A.edges_data = e_data

        for i in range(2):
            with tf.variable_scope(f"Layer{i}"):
                self.A.update(point_fn, edge_fn, None, ["UpdatePoint", "UpdateEdge", "UpdateGlobal"],
                              use_global_attr=False)

    def __call__(self):
        return self.A.points_data

@ROI_BOX_HEAD_REGISTRY.register()
class BBDNetX7(WModule):
    def __init__(self,cfg=None,parent=None,*args,**kwargs):
        super().__init__(cfg=cfg,parent=parent,*args,**kwargs)

    def forward(self,boxes_features):
        boxes = self.parent.t_proposal_boxes
        B,N,_ = wmlt.combined_static_and_dynamic_shape(boxes)
        BN,H,W,C = wmlt.combined_static_and_dynamic_shape(boxes_features)
        boxes_features = tf.reshape(boxes_features,[B,N,H,W,C])
        res = wmlt.static_or_dynamic_map_fn(lambda x:self.process_one_image(x[0],x[1]),elems=[boxes_features,boxes],
                        dtype=[tf.float32])
        B,N,C = wmlt.combined_static_and_dynamic_shape(res)
        res = tf.reshape(res,[B*N,C])
        return res

    def process_one_image(self,boxes_features,bboxes):
        bbd_net = BBDNetForOneImg(self.cfg,
                                bboxes,
                                boxes_features)
        return bbd_net()
