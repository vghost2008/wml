#coding=utf-8
from object_detection2.standard_names import *
from wmodule import WModule
from basic_tftools import batch_gather
import numpy as np
import object_detection2.bboxes as odb
import img_utils as wmli
import basic_tftools as btf


'''
image_data:[h,w,c]
bboxes:[N,4] absolute coordinate
rect:[ymin,xmin,ymax,xmax) absolute coordinate
'''
def cut_bboxes(bboxes,labels,img,rect,threshold=0.5,fill_color=None,is_sub_img=False):
    res_bboxes = []
    res_labels = []

    if not isinstance(labels,np.ndarray):
        labels = np.array(labels)

    remove_bboxes = []
    no_zero = 1e-3
    for i in range(labels.shape[0]):
        iou = odb.npbboxes_intersection_of_box0([bboxes[i]],rect)
        if iou<threshold and iou>no_zero:
            remove_bboxes.append(bboxes[i])
        elif iou>=threshold:
            res_bboxes.append(bboxes[i])
            res_labels.append(labels[i])

    if not is_sub_img:
        img = wmli.sub_image(img,rect)

    if fill_color is not None and len(remove_bboxes)>0:
        remove_bboxes = np.stack(remove_bboxes, axis=0) - np.array([[rect[0], rect[1], rect[0], rect[1]]])
        remove_bboxes = remove_bboxes.astype(np.int32)
        img = wmli.remove_boxes_of_img(img,remove_bboxes,default_value=fill_color)

    if len(res_labels)>0:
        res_bboxes = np.stack(res_bboxes,axis=0) - np.array([[rect[0],rect[1],rect[0],rect[1]]])
        res_labels = np.array(res_labels)
    else:
        res_bboxes = np.zeros(shape=[0,4],dtype=bboxes.dtype)
        res_labels = np.zeros(shape=[0],dtype=labels.dtype)

    return res_bboxes,res_labels,img

'''
在每一个标目标附近裁剪出一个子图
bboxes: [N,4] absolute coordinate
size:[H,W]
return:
[N,4] (ymin,xmin,ymax,xmax) absolute coordinate
'''
def get_random_cut_bboxes_rect(bboxes,size,img_size):
    res = []
    y_max,x_max = img_size[0],img_size[1]
    if not isinstance(bboxes,np.ndarray):
        bboxes = np.array(bboxes)
    if bboxes.shape[0] == 0:
        return []
    obj_ann_bboxes = odb.expand_bbox_by_size(bboxes,[x//2 for x in size],format='yxminmax')
    obj_ann_bboxes = odb.to_xyminwh(obj_ann_bboxes)


    for t_bbox in obj_ann_bboxes:
        t_bbox = list(t_bbox)
        t_bbox[1] = max(0,min(t_bbox[1],y_max))
        t_bbox[0] = max(0,min(t_bbox[0],x_max))
        t_bbox = odb.random_bbox_in_bbox(t_bbox,size)
        rect = (t_bbox[1],t_bbox[0],t_bbox[1]+t_bbox[3],t_bbox[0]+t_bbox[2])

        res.append(rect)
    return res

'''
在每一个标目标附近裁剪出一个子图, 如果一个bbox已经出现在前面的某一个rect中则跳过
用于保证一个instance仅在结果中出现一次
bboxes: [N,4] absolute coordinate
size:[H,W]
return:
[N,4] (ymin,xmin,ymax,xmax) absolute coordinate
'''
def get_random_cut_bboxes_rectv2(bboxes,size,img_size,labels=None,force_cut_labels=None):
    res = []
    y_max,x_max = img_size[0],img_size[1]
    if not isinstance(bboxes,np.ndarray):
        bboxes = np.array(bboxes)
    if bboxes.shape[0] == 0:
        return []
    obj_ann_bboxes = odb.expand_bbox_by_size(bboxes,[x//2 for x in size],format='yxminmax')
    obj_ann_bboxes = odb.to_xyminwh(obj_ann_bboxes)


    for i,t_bbox in enumerate(obj_ann_bboxes):
        bbox = bboxes[i]
        if len(res)>0:
            if labels is None or (labels[i] not in force_cut_labels):
                ious = odb.npbboxes_intersection_of_box0(bbox,res)
                max_index = np.argmax(ious)
                if ious[max_index]>0.9:
                    continue
        t_bbox = list(t_bbox)
        t_bbox[1] = max(0,min(t_bbox[1],y_max))
        t_bbox[0] = max(0,min(t_bbox[0],x_max))
        t_bbox = odb.random_bbox_in_bbox(t_bbox,size)
        rect = (t_bbox[1],t_bbox[0],t_bbox[1]+t_bbox[3],t_bbox[0]+t_bbox[2])

        res.append(rect)
    return res

def filter_by_classeswise_thresholds(labels,bboxes,probs,thresholds):
    '''

    :param labels: [N]
    :param bboxes: [N,4]
    :param probs: [N]
    :param thresholds: 不包含背景0
    :return:
    '''
    n_labels = []
    n_bboxes = []
    n_probs = []

    for i,l in enumerate(labels):
        tp = thresholds[l-1]
        p = probs[i]
        if tp<=p:
            n_labels.append(l)
            n_bboxes.append(bboxes[i])
            n_probs.append(p)

    return np.array(n_labels),np.array(n_bboxes),np.array(n_probs)
