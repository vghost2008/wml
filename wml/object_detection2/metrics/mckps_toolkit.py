import numpy as np
from .common import *
import sys
from .build import METRICS_REGISTRY
from wml.object_detection2.keypoints import mckps_distance_matrix
import math
import copy
'''
gtkps:[N,2]
gtlabels:[N]
kps: [M]
labels: [M]
scores: [M]
'''
def getMCKpsPrecision(gtkps,gtlabels,kps,labels,sigma=3,ext_info=False,is_crowd=None):
    if not isinstance(gtkps,np.ndarray):
        gtkps = np.array(gtkps)
    if not isinstance(gtlabels,np.ndarray):
        gtlabels = np.array(gtlabels)
    if is_crowd is None:
        is_crowd = np.zeros([gtlabels.shape[0]],dtype=bool)
    if not isinstance(is_crowd,np.ndarray):
        is_crowd = np.array(is_crowd)
    
    if kps.size == 0:
        if gtkps.size == 0:
            return 100.0,100.0
        return 100.0,0.0
    elif gtkps.size == 0:
        return 0.0,100.0

    gt_shape = gtkps.shape
    #indict if there have some kps match with this ground-truth kps
    gt_mask = np.zeros([gt_shape[0]],dtype=np.int32)
    kps_shape = kps.shape
    #indict if there have some ground-truth kps match with this kps
    kps_mask = np.zeros(kps_shape[0],dtype=np.int32)
    gt_size = gtlabels.shape[0]
    kps_size = labels.shape[0]
    dis_m = mckps_distance_matrix(gtkps,kps)
    for i in range(gt_size):

        cur_dis = dis_m[i]
        idxs = np.argsort(cur_dis)
        for idx in idxs:
            if kps_mask[idx] or gtlabels[i] != labels[idx]:
                continue
            cur_d = cur_dis[idx]
            if cur_d > sigma:
                break
            gt_mask[i] = 1
            kps_mask[idx] = 1
            break

    r_gt_mask = np.logical_or(gt_mask,is_crowd)
    correct_gt_num = np.sum(r_gt_mask)
    correct_bkps_num = np.sum(kps_mask)

    recall = safe_persent(correct_gt_num,gt_size)
    precision = safe_persent(correct_bkps_num,kps_size)
    P_v = gt_size
    TP_v = correct_bkps_num
    FP_v = kps_size-correct_bkps_num


    if ext_info:
        gt_label_list = []
        for i in range(gt_mask.shape[0]):
            if gt_mask[i] != 1:
                gt_label_list.append(gtlabels[i])
        pred_label_list = []
        for i in range(kps_size):
            if kps_mask[i] != 1:
                pred_label_list.append(labels[i])
        return precision,recall,gt_label_list,pred_label_list,TP_v,FP_v,P_v
    else:
        return precision,recall

'''
gtkps:[N,2]
gtlabels:[N]
kps: [M]
labels: [M]
scores: [M]
'''
def getMCKpsAccuracy(gtkps,gtlabels,kps,labels,sigma=3,ext_info=False,is_crowd=None):
    if not isinstance(gtkps,np.ndarray):
        gtkps = np.array(gtkps)
    if not isinstance(gtlabels,np.ndarray):
        gtlabels = np.array(gtlabels)
    if is_crowd is None:
        is_crowd = np.zeros([gtlabels.shape[0]],dtype=bool)
    if not isinstance(is_crowd,np.ndarray):
        is_crowd = np.array(is_crowd)
    
    if kps.size == 0:
        if gtkps.size == 0:
            return 100.0
        return 0.0
    elif gtkps.size == 0:
        return 0.0

    gt_shape = gtkps.shape
    #indict if there have some kps match with this ground-truth kps
    gt_mask = np.zeros([gt_shape[0]],dtype=np.int32)
    kps_shape = kps.shape
    #indict if there have some ground-truth kps match with this kps
    kps_mask = np.zeros(kps_shape[0],dtype=np.int32)
    gt_size = gtlabels.shape[0]
    kps_size = labels.shape[0]
    dis_m = mckps_distance_matrix(gtkps,kps)
    for i in range(gt_size):

        cur_dis = dis_m[i]
        idxs = np.argsort(cur_dis)
        for idx in idxs:
            if kps_mask[idx] or gtlabels[i] != labels[idx]:
                continue
            cur_d = cur_dis[idx]
            if cur_d > sigma:
                break
            gt_mask[i] = 1
            kps_mask[idx] = 1
            break

    r_gt_mask = np.logical_or(gt_mask,is_crowd)
    correct_gt_num = np.sum(r_gt_mask)
    #correct_bkps_num = np.sum(kps_mask)
    all_num = gt_size+kps_size-correct_gt_num

    acc = safe_persent(correct_gt_num,all_num)

    return acc


'''
gtkps:[N,2]
gtlabels:[N]
kps: [M]
labels: [M]
scores: [M]
return:
mAP:[0,100]
'''
def getKpsmAP(gtkps,gtlabels,kps,labels,scores=None,sigma=3,is_crowd=None):

    if not isinstance(gtkps,np.ndarray):
        gtkps = np.array(gtkps)
    if not isinstance(gtlabels,np.ndarray):
        gtlabels = np.array(gtlabels)
    if not isinstance(kps,np.ndarray):
        kps = np.array(kps)
    if not isinstance(labels,np.ndarray):
        labels = np.array(labels)
    if is_crowd is None:
        is_crowd = np.zeros([gtlabels.shape[0]],dtype=bool)
    if not isinstance(is_crowd,np.ndarray):
        is_crowd = np.array(is_crowd)
    gtkps = copy.deepcopy(np.array(gtkps))
    gtlabels = copy.deepcopy(np.array(gtlabels))
    kps = copy.deepcopy(kps)
    labels = copy.deepcopy(labels)
    if scores is not None:
        #按scores从小到大排列
        scores = copy.deepcopy(scores)
        index = np.argsort(scores)
        kps = kps[index]
        labels = labels[index]

    max_nr = 20
    data_nr = kps.shape[0]

    if data_nr==0:
        if gtkps.size == 0:
            return 100.0
        return 0.0

    if data_nr>max_nr:
        beg_index = range(0,data_nr,data_nr//max_nr)
    else:
        beg_index = range(0,data_nr)

    t_res = []

    for v in beg_index:
        p,r = getMCKpsPrecision(gtkps,gtlabels,kps[v:],labels[v:],sigma,is_crowd=is_crowd)
        t_res.append([p,r]) #r从大到小


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

class BaseMCKpsMetrics(BaseMetrics):
    def __init__(self,sigma=5,*args,**kwargs):
        super().__init__()
        self.sigma = sigma
        self.all_gt_keypoints = []
        self.all_gt_labels = []
        self.all_keypoints = []
        self.all_labels = []
        self.all_scores = []
        self.npall_gt_keypoints = None
        self.npall_gt_labels = None
        self.npall_keypoints = None
        self.npall_labels = None
        self.npall_scores = None
        self.offset = 0
        self.img_id = 0
        self.value = None
        print(f"Sigma={self.sigma}")

    '''
    gtkeypoitns:[N,2]
    gtlabels:[N]
    kps: [M,2]
    labels: [M]
    scores: [M]
    '''
    def __call__(self, gtkps,gtlabels,kps,labels,scores=None,area=None,iscrowd=None,probability=None):
        if probability is not None and scores is None:
            scores = probability
        c_offset = max(np.max(gtkps) if gtkps.size>0 else 0,np.max(kps) if kps.size>0 else 0)
        c_offset += self.sigma+1
        self.all_gt_keypoints.append(gtkps+self.offset)
        self.all_gt_labels.append(gtlabels)
        self.all_keypoints.append(kps+self.offset)
        self.all_labels.append(labels)
        self.all_scores.append(scores)
        self.offset += c_offset
        self.img_id += 1

    def evaluate(self):
        self.npall_gt_keypoints = np.concatenate(self.all_gt_keypoints,axis=0)
        self.npall_gt_labels = np.concatenate(self.all_gt_labels,axis=0)
        self.npall_keypoints = np.concatenate(self.all_keypoints,axis=0)
        self.npall_labels = np.concatenate(self.all_labels,axis=0)
        self.npall_scores = np.concatenate(self.all_scores,axis=0)

    def num_examples(self):
        return self.img_id

    def show(self,name=""):
        if len(name)>0:
            print(name)
        res = self.to_string()
        print(res)
        return res

    def to_string(self):
        if self.value is None:
            self.evaluate()
        return str(self.value)

@METRICS_REGISTRY.register()
class MCKpsMap(BaseMCKpsMetrics):
    def evaluate(self):
        super().evaluate()
        self.value = getKpsmAP(gtkps=self.npall_gt_keypoints,
                        gtlabels=self.npall_gt_labels,
                        kps=self.npall_keypoints,
                        labels=self.npall_labels,
                        scores=self.npall_scores,
                        sigma=self.sigma)
        return self.value

@METRICS_REGISTRY.register()
class MCKpsPrecisionAndRecall(BaseMCKpsMetrics):
    def __init__(self,threshold=None,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.threshold = threshold
        self.acc = None

    def evaluate(self):
        super().evaluate()
        if self.threshold is not None:
            keep = self.npall_scores>=self.threshold
            self.npall_scores = self.npall_scores[keep]
            self.npall_labels = self.npall_labels[keep]
            self.npall_keypoints = self.npall_keypoints[keep]
        self.value = getMCKpsPrecision(gtkps=self.npall_gt_keypoints,
                        gtlabels=self.npall_gt_labels,
                        kps=self.npall_keypoints,
                        labels=self.npall_labels,
                        #scores=self.npall_scores,
                        sigma=self.sigma)
        self.acc = getMCKpsAccuracy(gtkps=self.npall_gt_keypoints,
                        gtlabels=self.npall_gt_labels,
                        kps=self.npall_keypoints,
                        labels=self.npall_labels,
                        #scores=self.npall_scores,
                        sigma=self.sigma)
        self.p,self.r = self.value
        return self.value

    def to_string(self):
        if self.value is None:
            self.evaluate()
        p,r = self.value
        f1 = 2*p*r/max(p+r,1e-6)
        return f"P={p:.2f}, R={r:.2f}, f1={f1:.2f}, acc={self.acc:.2f}"
