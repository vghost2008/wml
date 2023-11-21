#coding=utf-8
import numpy as np
import os
import object_detection2.npod_toolkit as npod
import object_detection2.bboxes as odb
import math
import logging
from thirdparty.odmetrics import coco_evaluation
from thirdparty.odmetrics import standard_fields
import copy
import sys
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import wml_utils as wmlu
from .build import METRICS_REGISTRY
from abc import ABCMeta, abstractclassmethod

class BaseMetrics(metaclass=ABCMeta):
    def __init__(self) -> None:
        self._current_info = ""
        pass

    def current_info(self):
        return self._current_info

    def __repr__(self):
        return self.to_string()

    @abstractclassmethod
    def show(self):
        pass
    

def __safe_persent(v0,v1):
    if v1==0:
        return 100.
    else:
        return v0*100./v1

def getF1(gtboxes,gtlabels,boxes,labels,threshold=0.5):
    gt_shape = gtboxes.shape
    #indict if there have some box match with this ground-truth box
    gt_mask = np.zeros([gt_shape[0]],dtype=np.int32)
    boxes_shape = boxes.shape
    #indict if there have some ground-truth box match with this box
    boxes_mask = np.zeros(boxes_shape[0],dtype=np.int32)
    gt_size = gtlabels.shape[0]
    boxes_size = labels.shape[0]
    for i in range(gt_size):
        max_index = -1
        max_jaccard = 0.0
        #iterator on all boxes to find one which have the most maximum jacard value with current ground-truth box
        for j in range(boxes_size):
            if gtlabels[i] != labels[j] or boxes_mask[j] != 0:
                continue
            jaccard = npod.box_jaccard(gtboxes[i],boxes[j])
            if jaccard>threshold and jaccard > max_jaccard:
                max_jaccard = jaccard
                max_index = j

        if max_index < 0:
            continue

        gt_mask[i] = 1
        boxes_mask[max_index] = 1

    correct_num = np.sum(gt_mask)
    f1 = __safe_persent(2*correct_num,correct_num+gt_shape[0])

    return f1

'''
gtboxes:[X,4](ymin,xmin,ymax,xmax) relative coordinates, ground truth boxes
gtlabels:[X] the labels for ground truth boxes
boxes:[Y,4](ymin,xmin,ymax,xmax) relative coordinates,predicted boxes
labels:[Y], the labels for predicted boxes
probability:[Y], the probability for boxes, if probability is none, assum the boxes's probability is ascending order
return:
mAP:[0,100]
'''
def getmAP(gtboxes,gtlabels,boxes,labels,probability=None,threshold=0.5,is_crowd=None):

    if not isinstance(gtboxes,np.ndarray):
        gtboxes = np.array(gtboxes)
    if not isinstance(gtlabels,np.ndarray):
        gtlabels = np.array(gtlabels)
    if not isinstance(boxes,np.ndarray):
        boxes = np.array(boxes)
    if not isinstance(labels,np.ndarray):
        labels = np.array(labels)
    if is_crowd is None:
        is_crowd = np.zeros([gtlabels.shape[0]],dtype=np.bool)
    if not isinstance(is_crowd,np.ndarray):
        is_crowd = np.array(is_crowd)
    gtboxes = copy.deepcopy(np.array(gtboxes))
    gtlabels = copy.deepcopy(np.array(gtlabels))
    boxes = copy.deepcopy(boxes)
    labels = copy.deepcopy(labels)
    if probability is not None:
        probability = copy.deepcopy(probability)
        index = np.argsort(probability)
        boxes = boxes[index]
        labels = labels[index]

    max_nr = 20
    data_nr = boxes.shape[0]

    if data_nr==0:
        return 0.0

    if data_nr>max_nr:
        beg_index = range(0,data_nr,data_nr//max_nr)
    else:
        beg_index = range(0,data_nr)

    t_res = []

    for v in beg_index:
        p,r = getPrecision(gtboxes,gtlabels,boxes[v:],labels[v:],threshold,is_crowd=is_crowd)
        t_res.append([p,r])


    t_res1 = []
    old_v = None
    for v in t_res:
        if old_v is not None and v[0]<old_v[0]:
            v[0] = old_v[0]
        t_res1.append(v)
        old_v = v

    res = []
    old_v = None
    for v in reversed(t_res1):
        if old_v is not None:
            if v[1]<old_v[1]:
                v[1] = old_v[1]
            if math.fabs(v[1]-old_v[1])<1e-3 and v[0]<old_v[0]:
                v[0] = old_v[0]
        res.append(v)
        old_v = v

    min_r = res[0][1]
    max_r = res[-1][1]
    logging.debug("mAP: max r {}, min r {}".format(max_r,min_r))

    if min_r > 1e-2:
        res = np.concatenate([np.array([[res[0][0],0.]]),res],axis=0)
    if max_r <100.0-1e-2:
        l_precisions = res[-1][0]
        l_recall = res[-1][1]
        t_precision = min(l_precisions*l_recall/100.0,l_precisions)
        res = np.concatenate([res,np.array([[t_precision,100.0]])])

    res = np.array(res)
    res = res.transpose()
    precisions = res[0]
    recall = res[1]
    new_r = np.arange(0.,100.01,10.).tolist()
    new_p = []
    for r in new_r:
        new_p.append(np.interp(r,recall,precisions))
    precisions = np.array(new_p)
    return np.mean(precisions)


def getRecall(gtboxes,gtlabels,boxes,labels,threshold=0.5):
    gt_shape = gtboxes.shape
    #indict if there have some box match with this ground-truth box
    gt_mask = np.zeros([gt_shape[0]],dtype=np.int32)
    boxes_shape = boxes.shape
    #indict if there have some ground-truth box match with this box
    boxes_mask = np.zeros(boxes_shape[0],dtype=np.int32)
    gt_size = gtlabels.shape[0]
    boxes_size = labels.shape[0]
    for i in range(gt_size):
        max_index = -1
        max_jaccard = 0.0
        #iterator on all boxes to find one have the most maximum jacard value with current ground-truth box
        for j in range(boxes_size):
            if gtlabels[i] != labels[j] or boxes_mask[j] != 0:
                continue
            jaccard = npod.box_jaccard(gtboxes[i],boxes[j])
            if jaccard>threshold and jaccard > max_jaccard:
                max_jaccard = jaccard
                max_index = j

        if max_index < 0:
            continue

        gt_mask[i] = 1
        boxes_mask[max_index] = 1

    correct_num = np.sum(gt_mask)
    total_num = gt_size

    if 0 == total_num:
        return 100.

    return 100.*correct_num/total_num

def getAccuracy(gtboxes,gtlabels,boxes,labels,threshold=0.5,auto_scale_threshold=True,ext_info=False,is_crowd=None):
    '''
    :param gtboxes: [N,4]
    :param gtlabels: [N]
    :param boxes: [M,4]
    :param labels: [M]
    :param threshold: nms_threshold,float
    :return: precision,recall float
    '''
    if not isinstance(gtboxes,np.ndarray):
        gtboxes = np.array(gtboxes)
    if not isinstance(gtlabels,np.ndarray):
        gtlabels = np.array(gtlabels)
    if is_crowd is None:
        is_crowd = np.zeros([gtlabels.shape[0]],dtype=np.bool)
    if not isinstance(is_crowd,np.ndarray):
        is_crowd = np.array(is_crowd)
    gt_shape = gtboxes.shape
    #indict if there have some box match with this ground-truth box
    gt_mask = np.zeros([gt_shape[0]],dtype=np.int32)
    boxes_shape = boxes.shape
    #indict if there have some ground-truth box match with this box
    boxes_mask = np.zeros(boxes_shape[0],dtype=np.int32)
    gt_size = gtlabels.shape[0]
    boxes_size = labels.shape[0]
    MIN_VOL = 0.005
    for i in range(gt_size):
        max_index = -1
        max_jaccard = 0.0

        t_threshold = threshold
        if auto_scale_threshold:
            #print(i,gtboxes,gtlabels)
            vol = npod.box_vol(gtboxes[i])
            if vol < MIN_VOL:
                t_threshold = vol*threshold/MIN_VOL
        #iterator on all boxes to find one have the most maximum jacard value with current ground-truth box
        for j in range(boxes_size):
            if gtlabels[i] != labels[j] or boxes_mask[j] != 0:
                continue

            jaccard = npod.box_jaccard(gtboxes[i],boxes[j])
            if jaccard>t_threshold and jaccard > max_jaccard:
                max_jaccard = jaccard
                max_index = j

        if max_index < 0:
            continue

        gt_mask[i] = 1
        boxes_mask[max_index] = 1

    r_gt_mask = np.logical_or(gt_mask,is_crowd)
    correct_gt_num = np.sum(r_gt_mask)
    correct_bbox_num = np.sum(boxes_mask)
    correct_num = np.sum(gt_mask)
    r_gt_size = gt_size-correct_gt_num+correct_bbox_num

    P_v = gt_size
    TP_v = correct_bbox_num
    FP_v = boxes_size-correct_num

    return __safe_persent(TP_v,r_gt_size+boxes_size-correct_bbox_num)

def getPrecision(gtboxes,gtlabels,boxes,labels,threshold=0.5,auto_scale_threshold=True,ext_info=False,is_crowd=None):
    '''
    :param gtboxes: [N,4]
    :param gtlabels: [N]
    :param boxes: [M,4]
    :param labels: [M]
    :param threshold: nms_threshold,float
    :return: precision,recall float
    '''
    if not isinstance(gtboxes,np.ndarray):
        gtboxes = np.array(gtboxes)
    if not isinstance(gtlabels,np.ndarray):
        gtlabels = np.array(gtlabels)
    if is_crowd is None:
        is_crowd = np.zeros([gtlabels.shape[0]],dtype=np.bool)
    if not isinstance(is_crowd,np.ndarray):
        is_crowd = np.array(is_crowd)
    gt_shape = gtboxes.shape
    #indict if there have some box match with this ground-truth box
    gt_mask = np.zeros([gt_shape[0]],dtype=np.int32)
    boxes_shape = boxes.shape
    #indict if there have some ground-truth box match with this box
    boxes_mask = np.zeros(boxes_shape[0],dtype=np.int32)
    gt_size = gtlabels.shape[0]
    boxes_size = labels.shape[0]
    MIN_VOL = 0.005
    #print(">>>>",gtboxes,gtlabels)
    for i in range(gt_size):
        max_index = -1
        max_jaccard = 0.0

        t_threshold = threshold
        if auto_scale_threshold:
            #print(i,gtboxes,gtlabels)
            vol = npod.box_vol(gtboxes[i])
            if vol < MIN_VOL:
                t_threshold = vol*threshold/MIN_VOL
        #iterator on all boxes to find one have the most maximum jacard value with current ground-truth box
        for j in range(boxes_size):
            if gtlabels[i] != labels[j] or boxes_mask[j] != 0:
                continue

            jaccard = npod.box_jaccard(gtboxes[i],boxes[j])
            if jaccard>t_threshold and jaccard > max_jaccard:
                max_jaccard = jaccard
                max_index = j

        if max_index < 0:
            continue

        gt_mask[i] = 1
        boxes_mask[max_index] = 1

    r_gt_mask = np.logical_or(gt_mask,is_crowd)
    correct_gt_num = np.sum(r_gt_mask)
    correct_bbox_num = np.sum(boxes_mask)

    recall = __safe_persent(correct_gt_num,gt_size)
    precision = __safe_persent(correct_bbox_num,boxes_size)
    P_v = gt_size
    TP_v = correct_bbox_num
    FP_v = boxes_size-correct_bbox_num


    if ext_info:
        gt_label_list = []
        for i in range(gt_mask.shape[0]):
            if gt_mask[i] != 1:
                gt_label_list.append(gtlabels[i])
        pred_label_list = []
        for i in range(boxes_size):
            if boxes_mask[i] != 1:
                pred_label_list.append(labels[i])
        return precision,recall,gt_label_list,pred_label_list,TP_v,FP_v,P_v
    else:
        return precision,recall

def getEasyPrecision(gtboxes,gtlabels,boxes,labels,threshold=0.05,auto_scale_threshold=True,ext_info=False):
    '''
    :param gtboxes: [N,4]
    :param gtlabels: [N]
    :param boxes: [M,4]
    :param labels: [M]
    :param threshold: nms_threshold,float
    :return: precision,recall float
    '''
    if not isinstance(gtboxes,np.ndarray):
        gtboxes = np.array(gtboxes)
    if not isinstance(gtlabels,np.ndarray):
        gtlabels = np.array(gtlabels)
    gt_shape = gtboxes.shape
    #indict if there have some box match with this ground-truth box
    gt_mask = np.zeros([gt_shape[0]],dtype=np.int32)
    boxes_shape = boxes.shape
    #indict if there have some ground-truth box match with this box
    boxes_mask = np.zeros(boxes_shape[0],dtype=np.int32)
    gt_size = gtlabels.shape[0]
    boxes_size = labels.shape[0]
    MIN_VOL = 0.005
    #print(">>>>",gtboxes,gtlabels)
    for i in range(gt_size):
        max_index = -1
        max_jaccard = 0.0

        t_threshold = threshold
        if auto_scale_threshold:
            #print(i,gtboxes,gtlabels)
            vol = npod.box_vol(gtboxes[i])
            if vol < MIN_VOL:
                t_threshold = vol*threshold/MIN_VOL
        #iterator on all boxes to find one have the most maximum jacard value with current ground-truth box
        for j in range(boxes_size):
            if gtlabels[i] != labels[j] or boxes_mask[j] != 0:
                continue

            jaccard = npod.box_jaccard(gtboxes[i],boxes[j])
            if jaccard>t_threshold and jaccard > max_jaccard:
                max_jaccard = jaccard
                max_index = j

        if max_index < 0:
            continue

        gt_mask[i] = 1
        boxes_mask[max_index] = 1
    
    pred_labels = set(labels[boxes_mask.astype(np.bool)].tolist())
    for j in range(boxes_size):
        if boxes_mask[j] != 0:
            continue
        if labels[j] in pred_labels:
            boxes_mask[j] = 1
        

    correct_num = np.sum(gt_mask)
    correct_num1 = np.sum(boxes_mask)

    recall = __safe_persent(correct_num,gt_size)
    precision = __safe_persent(correct_num1,boxes_size)
    P_v = gt_size
    TP_v = correct_num
    FP_v = boxes_size-correct_num1


    if ext_info:
        gt_label_list = []
        for i in range(gt_mask.shape[0]):
            if gt_mask[i] != 1:
                gt_label_list.append(gtlabels[i])
        pred_label_list = []
        for i in range(boxes_size):
            if boxes_mask[i] != 1:
                pred_label_list.append(labels[i])
        return precision,recall,gt_label_list,pred_label_list,TP_v,FP_v,P_v
    else:
        return precision,recall

def getPrecisionV2(gt_data,pred_data,pred_func,threshold,return_f1=False):
    '''
    :param gt_data: N objects
    :param pred_data: M object
    :param pred_func: float (*)(obj0,obj1) get the distance of two objects, distance greater or equal zero
    :return: precision,recall float
    '''
    NR_GT = len(gt_data)
    NR_PRED = len(pred_data)
    #indict if there have some box match with this ground-truth box
    gt_mask = np.zeros([NR_GT],dtype=np.int32)
    #indict if there have some ground-truth box match with this box
    pred_mask = np.zeros(NR_PRED,dtype=np.int32)
    for i in range(NR_GT):
        min_index = -1
        min_dis = 1e10

        #iterator on all boxes to find one have the most maximum jacard value with current ground-truth box
        for j in range(NR_PRED):
            if pred_mask[j] != 0:
                continue
            dis = pred_func(gt_data[i],pred_data[j])
            if dis<threshold and dis< min_dis:
                min_dis = dis
                min_index = j

        if min_index < 0:
            continue

        gt_mask[i] = 1
        pred_mask[min_index] = 1

    correct_num = np.sum(gt_mask)

    recall = __safe_persent(correct_num,NR_GT)
    precision = __safe_persent(correct_num,NR_PRED)

    if return_f1:
        f1 = __safe_persent(2*correct_num,NR_PRED+NR_GT)
        return precision,recall,f1

    return precision,recall

@METRICS_REGISTRY.register()
class Accuracy(BaseMetrics):
    def __init__(self,threshold=0.1,num_classes=90,label_trans=None,classes_begin_value=1,*args,**kwargs):
        self.threshold = threshold
        self.gtboxes = []
        self.gtlabels = []
        self.is_crowd = []
        self.boxes = []
        self.labels = []
        self.precision = None
        self.recall = None
        self.total_test_nr = 0
        self.num_classes = num_classes
        self.label_trans = label_trans
        self.bboxes_offset = np.zeros([1,4],dtype=np.float32)
        del classes_begin_value

    def __call__(self, gtboxes,gtlabels,boxes,labels,probability=None,img_size=[512,512],
                 gtmasks=None,
                 masks=None,is_crowd=None,use_relative_coord=True):
        if self.label_trans is not None:
            gtlabels = self.label_trans(gtlabels)
            labels = self.label_trans(labels)
        if gtboxes.shape[0]>0:
            self.gtboxes.append(np.array(gtboxes)+self.bboxes_offset)
            self.gtlabels.append(np.array(gtlabels))
            if is_crowd is None:
                is_crowd = np.zeros([gtlabels.shape[0]],dtype=np.bool)
            self.is_crowd.append(np.array(is_crowd))
        if boxes.shape[0]>0:
            self.boxes.append(np.array(boxes)+self.bboxes_offset)
            self.labels.append(np.array(labels))

        self.total_test_nr += 1

        t_bboxes = np.concatenate([gtboxes,boxes],axis=0)
        t_max = np.max(t_bboxes,axis=0)
        max_0 = t_max[2]
        max_1 = t_max[3]
        offset = np.array([[max_0,max_1,max_0,max_1]],dtype=np.float32)
        self.bboxes_offset = self.bboxes_offset+offset

    def evaluate(self):
        if self.total_test_nr==0 or len(self.boxes)==0 or len(self.labels)==0:
            self.precision,self.recall = 0,0
            return
        gtboxes = np.concatenate(self.gtboxes,axis=0)
        gtlabels = np.concatenate(self.gtlabels,axis=0)
        boxes = np.concatenate(self.boxes,axis=0)
        labels = np.concatenate(self.labels,axis=0)
        self.acc = getAccuracy(gtboxes, gtlabels, boxes, labels, threshold=self.threshold,
                                                  auto_scale_threshold=False,
                                                  ext_info=False,
                                                  is_crowd=self.is_crowd)
    def show(self,name=""):
        self.evaluate()
        res = f"{name}: total test nr {self.total_test_nr}, acc {self.acc:.3f}"
        print(res)

    def to_string(self):
        try:
            return f"{self.acc:.3f}({self.total_test_nr})"
        except:
            return "N.A."

@METRICS_REGISTRY.register()
class PrecisionAndRecall(BaseMetrics):
    def __init__(self,threshold=0.5,num_classes=90,label_trans=None,classes_begin_value=1,*args,**kwargs):
        self.threshold = threshold
        self.gtboxes = []
        self.gtlabels = []
        self.boxes = []
        self.labels = []
        self.is_crowd = []
        self.precision = None
        self.recall = None
        self.total_test_nr = 0
        self.num_classes = num_classes
        self.label_trans = label_trans
        self.bboxes_offset = np.zeros([1,4],dtype=np.float32)
        del classes_begin_value

    def __call__(self, gtboxes,gtlabels,boxes,labels,probability=None,img_size=[512,512],
                 gtmasks=None,
                 masks=None,is_crowd=None,use_relative_coord=True):

        if self.label_trans is not None:
            gtlabels = self.label_trans(gtlabels)
            labels = self.label_trans(labels)

        if gtboxes.shape[0]>0:
            self.gtboxes.append(gtboxes+self.bboxes_offset)
            self.gtlabels.append(np.array(gtlabels))
            self.is_crowd.append(np.array(is_crowd))

        if boxes.shape[0]>0:
            self.boxes.append(boxes+self.bboxes_offset)
            self.labels.append(np.array(labels))
        
        t_bboxes = np.concatenate([gtboxes,boxes],axis=0)
        t_max = np.max(t_bboxes,axis=0)
        max_0 = t_max[2]
        max_1 = t_max[3]
        offset = np.array([[max_0,max_1,max_0,max_1]],dtype=np.float32)
        self.bboxes_offset = self.bboxes_offset+offset

        self.total_test_nr += 1

        cur_precision,cur_recall = getPrecision(gtboxes=gtboxes,
                                                gtlabels=gtlabels,
                                                boxes=boxes,labels=labels,
                                                threshold=self.threshold,
                                                auto_scale_threshold=False,
                                                ext_info=False,
                                                is_crowd=is_crowd)
        self._current_info = f"precision={cur_precision}, recall={cur_recall}"

    def evaluate(self):
        if self.total_test_nr==0 or len(self.boxes)==0 or len(self.labels)==0:
            self.precision,self.recall = 0,0
            return
        if len(self.gtboxes) == 0:
            gtboxes = np.zeros([0,4],dtype=np.float32)
            gtlabels = np.zeros([0],dtype=np.int32)
            is_crowd = np.zeros([0],dtype=np.bool)
        else:
            gtboxes = np.concatenate(self.gtboxes,axis=0)
            gtlabels = np.concatenate(self.gtlabels,axis=0)
            is_crowd = np.concatenate(self.is_crowd,axis=0).astype(np.bool)
        boxes = np.concatenate(self.boxes,axis=0)
        labels = np.concatenate(self.labels,axis=0)
        self.precision,self.recall = getPrecision(gtboxes, gtlabels, boxes, labels, threshold=self.threshold,
                                                  auto_scale_threshold=False,
                                                  ext_info=False,
                                                  is_crowd=is_crowd)
    @property
    def f1(self):
        return 2*self.precision*self.recall/max(self.precision+self.recall,1e-8)

    def show(self,name=""):
        self.evaluate()
        res = f"{name}: {self}"
        print(res)

    def value(self):
        return self.f1

    def to_string(self):
        try:
            return f"{self.precision:.3f}/{self.recall:.3f}/{self.f1}/({self.total_test_nr})"
        except:
            return "N.A."

    def __repr__(self):
        res = f"total test nr {self.total_test_nr}, precision {self.precision:.3f}, recall {self.recall:.3f}, f1 {self.f1}"
        return res

@METRICS_REGISTRY.register()
class EasyPrecisionAndRecall(BaseMetrics):
    def __init__(self,threshold=0.05,num_classes=90,label_trans=None,classes_begin_value=1,*args,**kwargs):
        self.threshold = threshold
        self.gtboxes = []
        self.gtlabels = []
        self.boxes = []
        self.labels = []
        self.precision = None
        self.recall = None
        self.total_test_nr = 0
        self.num_classes = num_classes
        self.label_trans = label_trans
        del classes_begin_value

    def __call__(self, gtboxes,gtlabels,boxes,labels,probability=None,img_size=[512,512],
                 gtmasks=None,
                 masks=None,is_crowd=None,use_relative_coord=True):
        if self.label_trans is not None:
            gtlabels = self.label_trans(gtlabels)
            labels = self.label_trans(labels)
        if gtboxes.shape[0]>0:
            self.gtboxes.append(gtboxes)
            self.gtlabels.append(np.array(gtlabels)+self.total_test_nr*self.num_classes)
        if boxes.shape[0]>0:
            self.boxes.append(boxes)
            self.labels.append(np.array(labels)+self.total_test_nr*self.num_classes)
        self.total_test_nr += 1

    def evaluate(self):
        if self.total_test_nr==0 or len(self.boxes)==0 or len(self.labels)==0:
            self.precision,self.recall = 0,0
            return
        gtboxes = np.concatenate(self.gtboxes,axis=0)
        gtlabels = np.concatenate(self.gtlabels,axis=0)
        boxes = np.concatenate(self.boxes,axis=0)
        labels = np.concatenate(self.labels,axis=0)
        self.precision,self.recall = getEasyPrecision(gtboxes, gtlabels, boxes, labels, threshold=self.threshold,
                                                  auto_scale_threshold=False, ext_info=False)
    @property
    def f1(self):
        return 2*self.precision*self.recall/max(self.precision+self.recall,1e-8)

    def show(self,name=""):
        self.evaluate()
        res = f"{name}: {self}"
        print(res)

    def value(self):
        return self.f1

    def to_string(self):
        try:
            return f"{self.precision:.3f}/{self.recall:.3f}/{self.f1}/({self.total_test_nr})"
        except:
            return "N.A."

    def __repr__(self):
        res = f"total test nr {self.total_test_nr}, precision {self.precision:.3f}, recall {self.recall:.3f}, f1 {self.f1}"
        return res

@METRICS_REGISTRY.register()
class ImgLevelPrecisionAndRecall(BaseMetrics):
    def __init__(self,threshold=0.5,num_classes=90,label_trans=None,classes_begin_value=1,*args,**kwargs):
        self.threshold = threshold
        self.precision = None
        self.recall = None
        self.total_test_nr = 0
        self.num_classes = num_classes
        self.label_trans = label_trans
        del classes_begin_value
        self.tp = 0
        self.fn = 0
        self.fp = 0

    def __call__(self, gtboxes,gtlabels,boxes,labels,probability=None,img_size=[512,512],
                 gtmasks=None,
                 masks=None,is_crowd=None,use_relative_coord=True):
        if self.label_trans is not None:
            gtlabels = self.label_trans(gtlabels)
            labels = self.label_trans(labels)
        if not isinstance(gtlabels,np.ndarray):
            gtlabels = np.array(gtlabels)
        ori_gtlabels = gtlabels.copy()
        gtlabels = set(gtlabels)
        labels = set(labels)
        union_labels = gtlabels&labels
        tp = len(union_labels)
        false_negative = gtlabels-union_labels
        fn = len(false_negative)

        if is_crowd is not None:
            for l in false_negative:
                mask = ori_gtlabels==l
                t_is_crowd = is_crowd[mask]
                if np.all(t_is_crowd):
                    fn = fn-1

        fp = len(labels)-tp
        self.tp += tp
        self.fn += fn
        self.fp += fp
        self.total_test_nr += 1
        if fp+tp>0:
            precision = tp/(fp+tp)
        else:
            precision = 1.0
        if fn+tp>0:
            recall = tp/(fn+tp)
        else:
            recall = 1.0
        self._current_info = f"precision={precision:.3f}, recall={recall:.3f}"

        

    def evaluate(self):
        if self.total_test_nr==0:
            self.precision,self.recall = 0,0
            return
        self.precision = self.tp/(self.fp+self.tp)
        self.recall = self.tp/(self.fn+self.tp)

    @property
    def f1(self):
        return 2*self.precision*self.recall/max(self.precision+self.recall,1e-8)

    def show(self,name=""):
        self.evaluate()
        res = f"{name}: {self}"
        print(res)

    def value(self):
        return self.f1

    def to_string(self):
        try:
            return f"{self.precision:.3f}/{self.recall:.3f}/{self.f1}/({self.total_test_nr})"
        except:
            return "N.A."

    def __repr__(self):
        res = f"total test nr {self.total_test_nr}, image level precision {self.precision:.3f}, recall {self.recall:.3f}, f1 {self.f1}"
        return res

@METRICS_REGISTRY.register()
class ROC:
    def __init__(self,threshold=0.5,num_classes=90,label_trans=None,classes_begin_value=1,*args,**kwargs):
        self.threshold = threshold
        self.gtboxes = []
        self.gtlabels = []
        self.boxes = []
        self.labels = []
        self.probs = []
        self.precision = None
        self.recall = None
        self.total_test_nr = 0
        self.num_classes = num_classes
        self.label_trans = label_trans
        self.results = None
        del classes_begin_value

    def __call__(self, gtboxes,gtlabels,boxes,labels,probability=None,img_size=[512,512],
                 gtmasks=None,
                 masks=None,is_crowd=None):
        if self.label_trans is not None:
            gtlabels = self.label_trans(gtlabels)
            labels = self.label_trans(labels)
        if gtboxes.shape[0]>0:
            self.gtboxes.append(gtboxes)
            self.gtlabels.append(np.array(gtlabels)+self.total_test_nr*self.num_classes)
        if boxes.shape[0]>0:
            self.boxes.append(boxes)
            self.labels.append(np.array(labels)+self.total_test_nr*self.num_classes)
            self.probs.append(np.array(probability))
        self.total_test_nr += 1

    def evaluate(self):
        if self.total_test_nr==0 or len(self.boxes)==0 or len(self.labels)==0:
            self.precision,self.recall = 0,0
            return
        gtboxes = np.concatenate(self.gtboxes,axis=0)
        gtlabels = np.concatenate(self.gtlabels,axis=0)
        boxes = np.concatenate(self.boxes,axis=0)
        labels = np.concatenate(self.labels,axis=0)
        probs = np.concatenate(self.probs,axis=0)
        self.results = []

        for p in np.arange(0,1,0.05):
            mask = np.greater(probs,p)
            t_boxes = boxes[mask]
            t_labels = labels[mask]
            precision, recall, gt_label_list, pred_label_list, TP_v, FP_v, P_v = \
                getPrecision(gtboxes, gtlabels, t_boxes, t_labels, threshold=self.threshold,
                                                  auto_scale_threshold=False, ext_info=True)
            self.results.append([p,precision,recall])

    def show(self,name=""):
        print(self.to_string())

    def to_string(self):
        self.evaluate()
        res = ""
        if self.results is None or len(self.results) == 0:
            return res
        for p, precision, recall in self.results:
            res += f"{p:.3f},{precision:.3f},{recall:.3f};\n"

        return res

class ModelPerformance:
    def __init__(self,threshold,no_mAP=False,no_F1=False):
        self.total_map = 0.
        self.total_recall = 0.
        self.total_precision = 0.
        self.total_F1 = 0.
        self.threshold = threshold
        self.test_nr = 0
        self.no_mAP=no_mAP
        self.no_F1 = no_F1

    def __call__(self, gtboxes,gtlabels,boxes,labels,probability=None):
        gtboxes = copy.deepcopy(np.array(gtboxes))
        gtlabels = copy.deepcopy(np.array(gtlabels))
        boxes = copy.deepcopy(boxes)
        labels = copy.deepcopy(labels)
        if probability is not None:
            probability = copy.deepcopy(probability)

        if self.no_mAP:
            ap = 0.
        else:
            ap = getmAP(gtboxes, gtlabels, boxes, labels, probability=probability,threshold=self.threshold)

        rc = getRecall(gtboxes, gtlabels, boxes, labels, self.threshold)

        if self.no_F1:
            f1 = 0.
        else:
            f1 = getF1(gtboxes, gtlabels, boxes, labels, self.threshold)

        pc,_ = getPrecision(gtboxes, gtlabels, boxes, labels, self.threshold)

        self.total_map += ap
        self.total_recall += rc
        self.total_precision  += pc
        self.total_F1 += f1
        self.test_nr += 1
        return ap,rc,pc,f1

    @staticmethod
    def safe_div(v0,v1):
        if math.fabs(v1)<1e-8:
            return 0.
        return v0/v1

    def __getattr__(self, item):
        if item=="mAP":
            return self.safe_div(self.total_map,self.test_nr)
        elif item =="recall":
            return self.safe_div(self.total_recall,self.test_nr)
        elif item=="precision":
            return self.safe_div(self.total_precision,self.test_nr)

class GeneralCOCOEvaluation(BaseMetrics):
    def __init__(self,categories_list=None,
                 num_classes=None,mask_on=False,label_trans=None,
                 classes_begin_value=1,
                 min_bbox_size=0):
        if categories_list is None:
            print(f"WARNING: Use default categories list, start classes is {classes_begin_value}")
            self.categories_list = [{"id":x+classes_begin_value,"name":str(x+classes_begin_value)} for x in range(num_classes)]
        else:
            self.categories_list = categories_list
        if not mask_on:
            self.coco_evaluator = coco_evaluation.CocoDetectionEvaluator(
                self.categories_list,include_metrics_per_category=False)
        else:
            self.coco_evaluator = coco_evaluation.CocoMaskEvaluator(
                self.categories_list,include_metrics_per_category=False)
        self.min_bbox_size = min_bbox_size
        if self.min_bbox_size > 0:
            print(f"{type(self).__name__}: set min_bbox_size to {self.min_bbox_size}")
        self.label_trans = label_trans
        self.image_id = 0
        self.cached_values = {}
    '''
    gtboxes:[N,4]
    gtlabels:[N]
    img_size:[H,W]
    gtmasks:[N,H,W]
    is_crowd: [N]
    '''
    def __call__(self, gtboxes,gtlabels,boxes,labels,probability=None,img_size=[512,512],
                 gtmasks=None,
                 masks=None,is_crowd=None,use_relative_coord=False):
        if self.min_bbox_size > 0:
            gtboxes = odb.clamp_bboxes(gtboxes,self.min_bbox_size)
            boxes = odb.clamp_bboxes(boxes,self.min_bbox_size)
        
        cur_ap = getmAP(gtboxes=gtboxes,
                        gtlabels=gtlabels,
                        boxes=boxes,
                        labels=labels,
                        probability=probability,
                        is_crowd=is_crowd)
        self._current_info = f"ap={cur_ap}"

        if probability is None:
            probability = np.ones_like(labels,dtype=np.float32)
        if not isinstance(gtboxes,np.ndarray):
            gtboxes = np.array(gtboxes)
        if not isinstance(gtlabels,np.ndarray):
            gtlabels = np.array(gtlabels)
        if not isinstance(boxes,np.ndarray):
            boxes = np.array(boxes)
        if not isinstance(labels,np.ndarray):
            labels = np.array(labels)
        if self.label_trans is not None:
            gtlabels = self.label_trans(gtlabels)
            labels = self.label_trans(labels)
        if probability is not None and not isinstance(probability,np.ndarray):
            probability = np.array(probability)
        if gtlabels.shape[0]>0:
            if use_relative_coord:
                gtboxes = gtboxes*[[img_size[0],img_size[1],img_size[0],img_size[1]]]
            groundtruth_dict={
                standard_fields.InputDataFields.groundtruth_boxes:
                    gtboxes,
                standard_fields.InputDataFields.groundtruth_classes:gtlabels,
            }
            if is_crowd is not None:
                if not isinstance(is_crowd,np.ndarray):
                    is_crowd = np.array(is_crowd)
                groundtruth_dict[standard_fields.InputDataFields.groundtruth_is_crowd] = is_crowd
            if gtmasks is not None:
                groundtruth_dict[standard_fields.InputDataFields.groundtruth_instance_masks] = gtmasks
            self.coco_evaluator.add_single_ground_truth_image_info(
                image_id=str(self.image_id),
                groundtruth_dict=groundtruth_dict)
        if labels.shape[0]>0 and gtlabels.shape[0]>0:
            if use_relative_coord:
                boxes = boxes*[[img_size[0],img_size[1],img_size[0],img_size[1]]]
            detections_dict={
                standard_fields.DetectionResultFields.detection_boxes:
                    boxes,
                standard_fields.DetectionResultFields.detection_scores:
                    probability,
                standard_fields.DetectionResultFields.detection_classes:
                    labels
            }
            if masks is not None:
                detections_dict[standard_fields.DetectionResultFields.detection_masks] = masks
            self.coco_evaluator.add_single_detected_image_info(
                image_id=str(self.image_id),
                detections_dict=detections_dict)
        self.image_id += 1

    def num_examples(self):
        if '_image_ids_with_detections' in self.coco_evaluator.__dict__:
            return len(self.coco_evaluator._image_ids_with_detections)
        elif '_image_ids' in self.coco_evaluator.__dict__:
            return len(self.coco_evaluator._image_ids)
        else:
            raise RuntimeError("Error evaluator type.")

    def evaluate(self):
        print(f"Test size {self.num_examples()}")
        return self.coco_evaluator.evaluate()

    def show(self,name=""):
        sys.stdout.flush()
        print(f"Test size {self.num_examples()}")
        res = self.coco_evaluator.evaluate()
        str0 = "|配置|"
        str1 = "|---|"
        str2 = f"|{name}|"
        for k,v in res.items():
            index = k.find("/")
            if index>0:
                k = k[index+1:]
            self.cached_values[k] = v
            str0 += f"{k}|"
            str1 += "---|"
            str2 += f"{v:.3f}|"
        print(str0)
        print(str1)
        print(str2)
        sys.stdout.flush()
        return res

    def to_string(self):
        if 'mAP' in self.cached_values and 'mAP@.50IOU' in self.cached_values:
            return f"{self.cached_values['mAP']:.3f}/{self.cached_values['mAP@.50IOU']:.3f}"
        else:
            return f"N.A."

    def __repr__(self):
        return self.to_string()
    
    def value(self):
        if 'mAP' in self.cached_values:
            return self.cached_values['mAP']
        elif  'mAP@.50IOU' in self.cached_values:
            return self.cached_values['mAP@.50IOU']
        else:
            return 0.0

@METRICS_REGISTRY.register()
class COCOBoxEvaluation(GeneralCOCOEvaluation):
    def __init__(self,categories_list=None,num_classes=None,label_trans=None,classes_begin_value=1,**kwargs):
        super().__init__(categories_list=categories_list,
                         num_classes=num_classes,
                         mask_on=False,
                         label_trans=label_trans,
                         classes_begin_value=classes_begin_value,
                         **kwargs)
@METRICS_REGISTRY.register()
class COCOMaskEvaluation(GeneralCOCOEvaluation):
    def __init__(self,categories_list=None,num_classes=None,label_trans=None,classes_begin_value=1,**kwargs):
        super().__init__(categories_list=categories_list,
                         num_classes=num_classes,
                         mask_on=True,
                         label_trans=label_trans,
                         classes_begin_value=classes_begin_value,
                         **kwargs)

@METRICS_REGISTRY.register()
class COCOEvaluation(BaseMetrics):
    '''
    num_classes: 不包含背景 
    '''
    def __init__(self,categories_list=None,num_classes=None,mask_on=False,label_trans=None,classes_begin_value=1,**kwargs):
        self.box_evaluator = COCOBoxEvaluation(categories_list=categories_list,
                                               num_classes=num_classes,
                                               label_trans=label_trans,
                                               classes_begin_value=classes_begin_value,
                                               **kwargs)
        self.mask_evaluator = None
        if mask_on:
            self.mask_evaluator = COCOMaskEvaluation(categories_list=categories_list,
                                                     num_classes=num_classes,
                                                     label_trans=label_trans,
                                                     classes_begin_value=classes_begin_value,
                                                     **kwargs)
    def __call__(self, *args, **kwargs):
        self.box_evaluator(*args,**kwargs)
        self._current_info = self.box_evaluator.current_info()
        if self.mask_evaluator is not None:
            self.mask_evaluator(*args,**kwargs)
            self._current_info += ", mask" + self.mask_evaluator.current_info()

    def num_examples(self):
        return self.box_evaluator.num_examples()

    def evaluate(self):
        res = self.box_evaluator.evaluate()
        if self.mask_evaluator is not None:
            res1 = self.mask_evaluator.evaluate()
            return res,res1
        return res

    def show(self,name=""):
        self.box_evaluator.show(name=name)
        if self.mask_evaluator is not None:
            self.mask_evaluator.show(name=name)

    def to_string(self):
        if self.mask_evaluator is not None:
            return self.box_evaluator.to_string()+";"+self.mask_evaluator.to_string()
        else:
            return self.box_evaluator.to_string()

@METRICS_REGISTRY.register()
class COCOKeypointsEvaluation(BaseMetrics):
    def __init__(self,num_joints,categories="person",oks_sigmas=None):
        categories_keypoints = []
        for i in range(num_joints):
            categories_keypoints.append({"id":i,"name":f"KP{i}"})
        self.coco_evaluator = coco_evaluation.CocoKeypointEvaluator(
                category_id=1,
                category_keypoints=categories_keypoints,
                class_text=categories,
                oks_sigmas=oks_sigmas)
        self.image_id = 0
        self.cached_values = {}
    '''
    gtboxes:[N,4]
    img_size:[H,W]
    gtkeypoitns:[N,num_joints,3]
    boxes: [M,4]
    kps: [M,num_joints,3]
    '''
    def __call__(self, gtboxes,gtkeypoints,kps,scores,area=None,iscrowd=None):
        self.image_id += 1
        self.add_groundtruth(self.image_id,gtboxes,gtkeypoints,area=area,iscrowd=iscrowd)
        self.add_detection(self.image_id,kps,scores)

    def add_groundtruth(self, image_id,gtboxes,gtkeypoints,area=None,iscrowd=None):
        if not isinstance(gtboxes,np.ndarray):
            gtboxes = np.array(gtboxes)
        if not isinstance(gtkeypoints,np.ndarray):
            gtkeypoints = np.array(gtkeypoints)

        if gtboxes.shape[0]>0:
            groundtruth_dict={
                standard_fields.InputDataFields.groundtruth_boxes:
                    gtboxes,
                standard_fields.InputDataFields.groundtruth_classes:np.ones([gtboxes.shape[0]],dtype=np.int32),
            }
            if iscrowd is not None:
                if not isinstance(iscrowd,np.ndarray):
                    iscrowd = np.array(iscrowd)
                groundtruth_dict[standard_fields.InputDataFields.groundtruth_is_crowd] = iscrowd
            if area is not None:
                if not isinstance(area,np.ndarray):
                    area = np.array(area)
                groundtruth_dict[standard_fields.InputDataFields.groundtruth_area] = area 
            groundtruth_dict[standard_fields.InputDataFields.groundtruth_keypoints] = gtkeypoints[...,:2]
            groundtruth_dict[standard_fields.InputDataFields.groundtruth_keypoint_visibilities] = gtkeypoints[...,2]
            self.coco_evaluator.add_single_ground_truth_image_info(
                image_id=str(image_id),
                groundtruth_dict=groundtruth_dict)

    def add_detection(self, image_id,kps,scores):
        if not isinstance(kps,np.ndarray):
            kps = np.array(kps)
        if not isinstance(scores,np.ndarray):
            scores = np.array(scores)

        if kps.shape[0]>0:
            detections_dict={
                standard_fields.DetectionResultFields.detection_boxes:np.zeros([kps.shape[0],4],dtype=np.float32),
                standard_fields.DetectionResultFields.detection_scores: scores,
                standard_fields.DetectionResultFields.detection_keypoints: kps[...,:2],
                standard_fields.DetectionResultFields.detection_classes:np.ones([kps.shape[0]],dtype=np.int32),
            }
            self.coco_evaluator.add_single_detected_image_info(
                image_id=str(image_id),
                detections_dict=detections_dict)

    def num_examples(self):
        if '_image_ids_with_detections' in self.coco_evaluator.__dict__:
            return len(self.coco_evaluator._image_ids_with_detections)
        elif '_image_ids' in self.coco_evaluator.__dict__:
            return len(self.coco_evaluator._image_ids)
        else:
            raise RuntimeError("Error evaluator type.")

    def evaluate(self):
        print(f"Test size {self.num_examples()}")
        res = self.coco_evaluator.evaluate()
        for k,v in res.items():
            index = k.find("/")
            if index>0:
                k = k[index+1:]
            self.cached_values[k] = v
        return res

    def show(self,name=""):
        sys.stdout.flush()
        print(f"Test size {self.num_examples()}")
        res = self.coco_evaluator.evaluate()
        str0 = "|配置|"
        str1 = "|---|"
        str2 = f"|{name}|"
        for k,v in res.items():
            index = k.find("/")
            if index>0:
                k = k[index+1:]
            self.cached_values[k] = v
            str0 += f"{k}|"
            str1 += "---|"
            str2 += f"{v:.3f}|"
        print(str0)
        print(str1)
        print(str2)
        sys.stdout.flush()
        return res

    def to_string(self):
        if 'mAP' in self.cached_values and 'mAP@.50IOU' in self.cached_values:
            return f"{self.cached_values['mAP']:.3f}/{self.cached_values['mAP@.50IOU']:.3f}"
        else:
            return f"N.A."

class ClassesWiseModelPerformace(BaseMetrics):
    def __init__(self,num_classes,threshold=0.5,classes_begin_value=1,model_type=COCOEvaluation,model_args={},label_trans=None,
                 **kwargs):
        self.num_classes = num_classes
        self.clases_begin_value = classes_begin_value
        model_args['classes_begin_value'] = classes_begin_value
        self.data = []
        for i in range(self.num_classes):
            self.data.append(model_type(num_classes=num_classes,**model_args))
        self.mp = model_type(num_classes=num_classes,**model_args)
        self.label_trans = label_trans
        self.have_data = np.zeros([num_classes],dtype=np.bool)

    @staticmethod
    def select_bboxes_and_labels(bboxes,labels,classes):
        if len(labels) == 0:
            return np.array([],dtype=np.float32),np.array([],dtype=np.int32),np.array([],dtype=np.bool)
        if not isinstance(labels,np.ndarray):
            labels = np.array(labels)
        mask = np.equal(labels,classes)
        rbboxes = bboxes[mask,:]
        rlabels = labels[mask]
        return rbboxes,rlabels,mask

    def __call__(self, gtboxes,gtlabels,boxes,labels,probability=None,img_size=None,use_relative_coord=False,is_crowd=None):
        if not isinstance(gtboxes,np.ndarray):
            gtboxes = np.array(gtboxes)
        if not isinstance(gtlabels,np.ndarray):
            gtlabels = np.array(gtlabels)
        if not isinstance(labels,np.ndarray):
            labels = np.array(labels)
        if self.label_trans is not None:
            gtlabels = self.label_trans(gtlabels)
            labels = self.label_trans(labels)
        
        if is_crowd is not None and not isinstance(is_crowd,np.ndarray):
            is_crowd = np.array(is_crowd)
            
        for i in range(self.num_classes):
            classes = i+self.clases_begin_value
            lgtboxes,lgtlabels,lgtmask = self.select_bboxes_and_labels(gtboxes,gtlabels,classes)
            lboxes,llabels,lmask = self.select_bboxes_and_labels(boxes,labels,classes)
            if is_crowd is not None:
                lis_crowd = is_crowd[lgtmask]
            else:
                lis_crowd = None
            if probability is not None:
                lprobs = probability[lmask]
            else:
                lprobs = None
            if (lgtlabels.shape[0]==0) and (llabels.shape[0] ==0):
                continue
            self.have_data[i] = True
            self.data[i](lgtboxes,lgtlabels,lboxes,llabels,lprobs,img_size=img_size,use_relative_coord=use_relative_coord,is_crowd=lis_crowd)
        return self.mp(gtboxes,gtlabels,boxes,labels,probability,is_crowd=is_crowd,img_size=img_size,
                        use_relative_coord=use_relative_coord)

    def show(self):
        sys.stdout.flush()
        for i in range(self.num_classes):
            if not self.have_data[i]:
                continue
            classes = i+self.clases_begin_value
            print(f"Classes:{classes}")
            try:
                self.data[i].show()
            except:
                print("N.A.")
                pass
        self.classes_wise_results = {}
        sys.stdout.flush()
        print(f"---------------------------------------------------------------")
        print(f"All classes")
        sys.stdout.flush()
        self.mp.show()
        sys.stdout.flush()
        print(f"Per classes")
        str0 = "|配置|"
        str1 = "|---|"
        str2 = "||"
        for i in range(self.num_classes):
            classes_id = i+1
            str0 += f"C{i+1}|"
            str1 += "---|"
            str2 += f"{str(self.data[i].to_string())}|"
            try:
                keys = ['mAP', 'mAP (small)', 'mAP (medium)', 'mAP (large)']
                if hasattr(self.data[i], "box_evaluator"):
                    d = self.data[i].box_evaluator.cached_values
                else:
                    d = self.data[i].cached_values
                values = [d[k] for k in keys]
                self.classes_wise_results[classes_id] = values
            except:
                self.classes_wise_results[classes_id] = [-1.0]*len(keys)
        print(str0)
        print(str1)
        print(str2)
        #wmlu.show_dict(self.classes_wise_results,format="{:.3f}")
        print("Summary")
        wmlu.show_dict(self.classes_wise_results)
        sys.stdout.flush()
        return str2

    def __getattr__(self, item):
        if item=="mAP":
            return self.mp.mAP
        elif item =="recall":
            return self.mp.recall
        elif item=="precision":
            return self.mp.precision

class SubsetsModelPerformace(BaseMetrics):
    def __init__(self, num_classes, sub_sets,threshold=0.5, model_type=COCOEvaluation, classes_begin_value=1,model_args={},
                 label_trans=None,
                 **kwargs):
        '''

        :param num_classes: 不包含背景
        :param sub_sets: list(list):如[[1,2],[3,4,5]]表示label 1,2一组进行评估,label 3 ,4,5一组进行评估
        :param threshold:
        :param classes_begin_value:
        :param model_type:
        :param model_args:
        :param label_trans:
        '''
        model_args['classes_begin_value'] = classes_begin_value
        self.num_classes = num_classes
        self.data = []
        self.sub_sets = sub_sets
        for i in range(len(sub_sets)):
            self.data.append(model_type(num_classes=num_classes, **model_args))
        self.mp = model_type(num_classes=num_classes, **model_args)
        self.label_trans = label_trans

    @staticmethod
    def select_bboxes_and_labels(bboxes, labels, classes):
        if len(labels) == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int32), np.array([], dtype=np.bool)

        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        mask = np.zeros_like(labels, dtype=np.bool)
        for i,l in enumerate(labels):
            if l in classes:
                mask[i] = True
        rbboxes = bboxes[mask, :]
        rlabels = labels[mask]
        return rbboxes, rlabels,mask

    def __call__(self, gtboxes, gtlabels, boxes, labels, probability=None, img_size=None):
        if not isinstance(gtboxes, np.ndarray):
            gtboxes = np.array(gtboxes)
        if not isinstance(gtlabels, np.ndarray):
            gtlabels = np.array(gtlabels)
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        if self.label_trans is not None:
            gtlabels = self.label_trans(gtlabels)
            labels = self.label_trans(labels)

        for i,sub_set_labels in enumerate(self.sub_sets):
            lgtboxes, lgtlabels,_ = self.select_bboxes_and_labels(gtboxes, gtlabels, sub_set_labels)
            lboxes, llabels,lmask = self.select_bboxes_and_labels(boxes, labels, sub_set_labels)
            if probability is not None:
                lprobs = probability[lmask]
            else:
                lprobs = None
            if lgtlabels.shape[0] == 0:
                continue
            self.data[i](lgtboxes, lgtlabels, lboxes, llabels, lprobs,img_size=img_size)
        return self.mp(gtboxes, gtlabels, boxes, labels)

    def current_info(self):
        return self.mp.current_info()

    def show(self):
        for i,sub_set_labels in enumerate(self.sub_sets):
            print(f"Classes:{sub_set_labels}")
            self.data[i].show()
        str0 = "|配置|"
        str1 = "|---|"
        str2 = "||"
        for i,sub_set_labels in enumerate(self.sub_sets):
            str0 += f"S{sub_set_labels}|"
            str1 += "---|"
            str2 += f"{str(self.data[i].to_string())}|"
        print(str0)
        print(str1)
        print(str2)

    def __getattr__(self, item):
        if item == "mAP":
            return self.mp.mAP
        elif item == "recall":
            return self.mp.recall
        elif item == "precision":
            return self.mp.precision

@METRICS_REGISTRY.register()
class  MeanIOU(BaseMetrics):
    def __init__(self,num_classes,*args,**kwargs):
        self.intersection = np.zeros(shape=[num_classes],dtype=np.int64)
        self.union = np.zeros(shape=[num_classes],dtype=np.int64)
        self.num_classes = num_classes

    def get_per_classes_iou(self):
        return self.intersection/np.maximum(self.union,1e-8)

    def get_mean_iou(self):
        return np.mean(self.get_per_classes_iou())
        
    
    def __call__(self, gtlabels,predictions):
        all_equal = np.equal(gtlabels,predictions)
        for i in range(1,self.num_classes+1):
            mask = np.equal(gtlabels,i)
            t_int = np.sum(all_equal[mask].astype(np.int64))
            t_data0 = np.sum(np.equal(gtlabels,i).astype(np.int64))
            t_data1 = np.sum(np.equal(predictions,i).astype(np.int64))
            t_union = t_data0+t_data1-t_int
            self.intersection[i-1] += t_int
            self.union[i-1] += t_union


    def show(self,name):
        str0 = "|配置|mIOU|"
        str1 = "|---|---|"
        str2 = f"|{name}|{self.get_mean_iou():.4f}"
        data = self.get_per_classes_iou()
        for i in range(self.num_classes):
            str0 += f"C{i+1}|"
            str1 += "---|"
            str2 += f"{data[i]:.3f}|"
        print(str0)
        print(str1)
        print(str2)

def coco_keypoint_eval_file(gt_file, res_file):
    coco = COCO(gt_file)
    coco_dt = coco.loadRes(res_file)
    coco_eval = COCOeval(coco, coco_dt, 'keypoints')
    coco_eval.params.useSegm = None
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']

    info_str = []
    for ind, name in enumerate(stats_names):
        info_str.append((name, coco_eval.stats[ind]))

    return info_str

def coco_bbox_eval_file(gt_file, res_file):
    coco = COCO(gt_file)
    coco_dt = coco.loadRes(res_file)
    coco_eval = COCOeval(coco, coco_dt, 'bbox')
    coco_eval.params.useSegm = None
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']

    info_str = []
    for ind, name in enumerate(stats_names):
        info_str.append((name, coco_eval.stats[ind]))

    return info_str

class WMAP(BaseMetrics):
    def __init__(self,categories_list=None,num_classes=None,mask_on=False,label_trans=None,classes_begin_value=1,threshold=0.5):
        if categories_list is None:
            print(f"WARNING: Use default categories list, start classes is {classes_begin_value}")
            self.categories_list = [{"id":x+classes_begin_value,"name":str(x+classes_begin_value)} for x in range(num_classes)]
        else:
            self.categories_list = categories_list
        self.label_trans = label_trans
        self.image_id = 0
        self.num_classes = num_classes
        self.cached_values = {}
        self.a_gtbboxes = []
        self.a_gtlabels = []
        self.a_bboxes = []
        self.a_labels = []
        self.a_scores = []
        self.threshold = 0.5
        self.map = None
    '''
    gtboxes:[N,4]
    gtlabels:[N]
    img_size:[H,W]
    gtmasks:[N,H,W]
    '''
    def __call__(self, gtboxes,gtlabels,boxes,labels,probability=None,img_size=[512,512],
                 gtmasks=None,
                 masks=None,is_crowd=None,use_relative_coord=False):
        if probability is None:
            probability = np.ones_like(labels,dtype=np.float32)
        if not isinstance(gtboxes,np.ndarray):
            gtboxes = np.array(gtboxes)
        if not isinstance(gtlabels,np.ndarray):
            gtlabels = np.array(gtlabels)
        if not isinstance(boxes,np.ndarray):
            boxes = np.array(boxes)
        if not isinstance(labels,np.ndarray):
            labels = np.array(labels)
        if self.label_trans is not None:
            gtlabels = self.label_trans(gtlabels)
            labels = self.label_trans(labels)
        if probability is not None and not isinstance(probability,np.ndarray):
            probability = np.array(probability)

        cur_ap = getmAP(gtboxes=gtboxes,
                        gtlabels=gtlabels,
                        boxes=boxes,
                        labels=labels,
                        probability=probability,
                        is_crowd=is_crowd)
        self._current_info = f"ap={cur_ap}"

        if gtlabels.shape[0]>0:
            if use_relative_coord:
                gtboxes = gtboxes*[[img_size[0],img_size[1],img_size[0],img_size[1]]]
            gtlabels = gtlabels+self.image_id*self.num_classes
            self.a_gtbboxes.append(gtboxes)
            self.a_gtlabels.append(gtlabels)
        if labels.shape[0]>0 and gtlabels.shape[0]>0:
            if use_relative_coord:
                boxes = boxes*[[img_size[0],img_size[1],img_size[0],img_size[1]]]
            labels = gtlabels+self.image_id*self.num_classes
            self.a_bboxes.append(boxes)
            self.a_labels.append(labels)
            self.a_scores.append(probability)
        self.image_id += 1

    def num_examples(self):
        return self.image_id

    def evaluate(self):
        print(f"Test size {self.num_examples()}")
        gtlabels = np.stack(self.a_gtbboxes,axis=0)
        gtbboxes = np.stack(self.a_gtbboxes,axis=0)
        labels = np.stack(self.a_labels,axis=0)
        bboxes = np.stack(self.a_bboxes,axis=0)
        scores = np.stack(self.a_scores,axis=0)
        self.map = getmAP(gtbboxes,gtlabels,bboxes,labels,scores,threshold=self.threshold)

    def show(self,name=""):
        sys.stdout.flush()
        print(f"Test size {self.num_examples()}")
        print(f"mAP={self.map}")
        return self.map

    def to_string(self):
        return self.map

    def __repr__(self):
        return self.to_string()
    
    def value(self):
        self.map

class ComposeMetrics(BaseMetrics):
    def __init__(self,*args,**kwargs):
        super().__init__()
        self.metrics = list(args)+list(kwargs.values())

    def __call__(self, *args,**kwargs):
        [m(*args,**kwargs) for m in self.metrics]
        self._current_info = "; ".join([m.current_info() for m in self.metrics])

    def evaluate(self):
        [m.evaluate() for m in self.metrics]

    def show(self,name=""):
        [m.show(name=name) for m in self.metrics]

    def to_string(self):
        return ";".join([m.to_string() for m in self.metrics])

    def value(self):
        return self.metrics[0].value()
