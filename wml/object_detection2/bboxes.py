#coding=utf-8
import numpy as np
import os
import sys
import random
from collections.abc import Iterable
import math
from .wmath import npsafe_divide
import cv2 as cv
import torch

'''
bbox_ref:[1,4]/[N,4], [[ymin,xmin,ymax,xmax]]
bboxes:[N,4],[[ymin,xmin,ymax,xmax],...]
return:
[N]
'''
def npbboxes_jaccard(bbox_ref, bboxes, name=None):

    bboxes = np.transpose(bboxes)
    bbox_ref = np.transpose(bbox_ref)
    int_ymin = np.maximum(bboxes[0], bbox_ref[0])
    int_xmin = np.maximum(bboxes[1], bbox_ref[1])
    int_ymax = np.minimum(bboxes[2], bbox_ref[2])
    int_xmax = np.minimum(bboxes[3], bbox_ref[3])
    h = np.maximum(int_ymax - int_ymin, 0.)
    w = np.maximum(int_xmax - int_xmin, 0.)
    inter_vol = h * w
    union_vol = -inter_vol \
                + (bboxes[2] - bboxes[0]) * (bboxes[3] - bboxes[1]) \
                + (bbox_ref[2] - bbox_ref[0]) * (bbox_ref[3] - bbox_ref[1])
    jaccard = npsafe_divide(inter_vol, union_vol, 'jaccard')
    return jaccard

'''
box0:[N,4], or [1,4],[ymin,xmin,ymax,xmax],...
box1:[N,4], or [1,4]
return:
[N],返回box0,box1交叉面积占box0的百分比
'''
def npbboxes_intersection_of_box0(box0,box1):

    bbox_ref= np.transpose(box0)
    bboxes = np.transpose(box1)
    int_ymin = np.maximum(bboxes[0], bbox_ref[0])
    int_xmin = np.maximum(bboxes[1], bbox_ref[1])
    int_ymax = np.minimum(bboxes[2], bbox_ref[2])
    int_xmax = np.minimum(bboxes[3], bbox_ref[3])
    h = np.maximum(int_ymax - int_ymin, 0.)
    w = np.maximum(int_xmax - int_xmin, 0.)
    inter_vol = h * w
    union_vol = (bbox_ref[2] - bbox_ref[0]) * (bbox_ref[3] - bbox_ref[1])
    union_vol = np.maximum(union_vol,1e-6)
    jaccard = inter_vol/union_vol
    return jaccard

'''
将以ymin,xmin,ymax,xmax表示的box转换为以cy,cx,size,ratio表示的box
'''
def to_cxysa(data):
    data = np.reshape(data,[-1,4])
    new_shape = data.shape
    res_data = np.zeros_like(data)
    for i in range(new_shape[0]):
        cy = (data[i][0]+data[i][2])*0.5
        cx= (data[i][1] + data[i][3]) * 0.5
        width = (data[i][3]-data[i][1])
        height = (data[i][2]-data[i][0])
        size = math.sqrt(width*height)
        if width>0.0:
            ratio = height/width
        else:
            ratio = 0
        res_data[i][0] = cy
        res_data[i][1] = cx
        res_data[i][2] = size
        res_data[i][3] = ratio

    return res_data

'''
data:[X,4] (ymin,xmin,ymax,xmax)
return:
[X,4] (cy,cx,h,w)
'''
def npto_cyxhw(data):
    data = np.transpose(data)
    ymin,xmin,ymax,xmax = data[0],data[1],data[2],data[3]
    cy = (ymin+ymax)/2
    cx = (xmin+xmax)/2
    h = ymax-ymin
    w = xmax-xmin
    data = np.stack([cy,cx,h,w],axis=1)
    return data

'''
data:[X,4] (cy,cx,h,w)
return:
[X,4] (ymin,xmin,ymax,xmax)
'''
def npto_yminxminymaxxmax(data):
    data = np.transpose(data)
    cy,cx,h,w= data[0],data[1],data[2],data[3]
    ymin = cy-h/2
    xmin = cx-w/2
    ymax = cy+h/2
    xmax = cx+w/2
    data = np.stack([ymin,xmin,ymax,xmax],axis=1)
    return data

'''
data:[X,4] (ymin,xmin,ymax,xmax)
return:
[X,4] (minx,miny,w,h)
'''
def npto_ctlwh(data):
    ymin,xmin,ymax,xmax = data[...,0],data[...,1],data[...,2],data[...,3]
    h = (ymax-ymin)
    w = (xmax-xmin)
    data = np.stack([xmin,ymin,w,h],axis=-1)
    return data

def npminxywh_toyxminmax(data):
    if not isinstance(data,np.ndarray):
        data = np.array(data)
    x = data[...,0]
    y = data[...,1]
    w = data[...,2]
    h = data[...,3]
    xmax = x+w
    ymax = y+h
    return np.stack([y,x,ymax,xmax],axis=-1)

'''
input:[4]/[N,4] [ymin,xmin,ymax,xmax]
output:[xmin,ymin,width,height]
'''
def to_xyminwh(bbox,is_absolute_coordinate=True):
    if not isinstance(bbox,np.ndarray):
        bbox = np.array(bbox)
    if len(bbox.shape)>= 2:
        ymin = bbox[...,0]
        xmin = bbox[...,1]
        ymax = bbox[...,2]
        xmax = bbox[...,3]
        w = xmax-xmin
        h = ymax-ymin
        return np.stack([xmin,ymin,w,h],axis=-1)

    else:
        if is_absolute_coordinate:
            return (bbox[1],bbox[0],bbox[3]-bbox[1]+1,bbox[2]-bbox[0]+1)
        else:
            return (bbox[1],bbox[0],bbox[3]-bbox[1],bbox[2]-bbox[0])

'''
boxes:[...,4] ymin,xmin,ymax,xmax
scale:[hscale,wscale]
max_size: [H,W]
缩放后中心点位置不变
'''
def npscale_bboxes(bboxes,scale,correct=False,max_size=None):
    if not isinstance(scale,Iterable):
        scale = [scale,scale]
    ymin,xmin,ymax,xmax = bboxes[...,0],bboxes[...,1],bboxes[...,2],bboxes[...,3]
    cy = (ymin+ymax)/2.
    cx = (xmin+xmax)/2.
    h = ymax-ymin
    w = xmax-xmin
    h = scale[0]*h
    w = scale[1]*w
    ymin = cy - h / 2.
    ymax = cy + h / 2.
    xmin = cx - w / 2.
    xmax = cx + w / 2.
    xmin = np.maximum(xmin,0)
    ymin = np.maximum(ymin,0)
    if max_size is not None:
        xmax = np.minimum(xmax,max_size[1]-1)
        ymax = np.minimum(ymax,max_size[0]-1)
    data = np.stack([ymin, xmin, ymax, xmax], axis=-1)
    return data

'''
boxes:[...,4] ymin,xmin,ymax,xmax
scale:[hscale,wscale]
'''
def npclip_bboxes(bboxes,max_size):
    ymin,xmin,ymax,xmax = bboxes[...,0],bboxes[...,1],bboxes[...,2],bboxes[...,3]
    xmin = np.maximum(xmin,0)
    ymin = np.maximum(ymin,0)
    if max_size is not None:
        xmax = np.minimum(xmax,max_size[1]-1)
        ymax = np.minimum(ymax,max_size[0]-1)
    data = np.stack([ymin, xmin, ymax, xmax], axis=-1)
    return data

def npchangexyorder(bboxes):
    if len(bboxes)==0:
        return bboxes
    bboxes = np.array(bboxes)
    ymin,xmin,ymax,xmax = bboxes[...,0],bboxes[...,1],bboxes[...,2],bboxes[...,3]
    data = np.stack([xmin, ymin, xmax, ymax], axis=-1)
    return data

'''
data:[N,4]
校正ymin,xmin,ymax,xmax表示的box中的不合法数据
'''
def correct_yxminmax_boxes(data,keep_size=False):
    if not keep_size:
        data = np.minimum(data,1.)
        data = np.maximum(data,0.)
    else:
        nr = data.shape[0]
        for i in range(nr):
            if data[i][0]<0.:
                data[i][2] -= data[i][0]
                data[i][0] = 0.
            if data[i][1]<0.:
                data[i][3] -= data[i][1]
                data[i][1] = 0.

            if data[i][2] > 1.:
                data[i][0] -= (data[i][2]-1.)
                data[i][2] = 1.
            if data[i][3] > 1.:
                data[i][1] -= (data[i][3]-1.)
                data[i][3] = 1.

    return data


'''
获取bboxes在box中的相对坐标
如:box=[5,5,10,10,]
bboxes=[[7,7,8,9]]
return:
[[2,2,3,4]]
'''
def get_boxes_relative_to_box(box,bboxes,remove_zero_size_box=False):
    if not isinstance(bboxes,np.ndarray):
        bboxes = np.array(bboxes)
    if bboxes.shape[0] == 0:
        return bboxes
    ymin,xmin,ymax,xmax = np.transpose(bboxes,[1,0])
    ymin = ymin-box[0]
    xmin = xmin-box[1]
    ymax = ymax-box[0]
    xmax = xmax-box[1]
    if remove_zero_size_box:
        mask = np.logical_and((ymax-ymin)>0,(xmax-xmin)>0)
        ymin = ymin[mask]
        xmin = xmin[mask]
        ymax = ymax[mask]
        xmax = xmax[mask]
    bboxes = np.stack([ymin,xmin,ymax,xmax],axis=1)
    return bboxes


'''
cnt:[[x,y],[x,y],...]
return the bbox of a contour
'''
def bbox_of_contour(cnt):
    all_points = np.array(cnt)
    points = np.transpose(all_points)
    x,y = np.vsplit(points,2)
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)
    return (ymin,xmin,ymax,xmax)

'''
bbox:(xmin,ymin,width,height)
'''
def random_int_in_bbox(bbox):
    x = random.randint(int(bbox[0]),int(bbox[0]+bbox[2]-1))
    y = random.randint(int(bbox[1]),int(bbox[1]+bbox[3]-1))
    return x,y

'''
bbox:(xmin,ymin,width,height)
size:(width,height) the size of return bbox
random return a box with center point in the input bbox
output:
[xmin,ymin,width,height]
'''
def random_bbox_in_bbox(bbox,size):
    x,y = random_int_in_bbox(bbox)
    xmin,ymin = max(0,x-size[0]//2),max(0,y-size[1]//2)
    return [xmin,ymin,size[0],size[1]]

'''
weights [2,x],[0] values,[1]:labels
bboxes:[N,4],[xmin,ymin,width,height]
'''
def random_bbox_in_bboxes(bboxes,size,weights=None,labels=None):
    if len(bboxes) == 0:
        return (0,0,size[0],size[1])
    if weights is not None:
        old_v = 0.0
        values = []

        for v in weights[0]:
            old_v += v
            values.append(old_v)
        random_v = random.uniform(0.,old_v)
        index = 0
        for i,v in enumerate(values):
            if random_v<v:
                index = i
                break
        label = weights[1][index]
        _bboxes = []
        for l,bbox in zip(labels,bboxes):
            if l==label:
                _bboxes.append(bbox)

        if len(_bboxes) == 0:
            return random_bbox_in_bboxes(bboxes,size)
        else:
            return random_bbox_in_bboxes(_bboxes,size)
    else:
        index = random.randint(0,len(bboxes)-1)
        return random_bbox_in_bbox(bboxes[index],size)

'''
bbox:[(xmin,ymin,width,height),....] (format="xyminwh") or [(ymin,xmin,ymax,xmax),...] (format="yxminmax")
return a list of new bbox with the size scale times of the input
'''
def expand_bbox(bboxes,scale=2,format="xyminwh"):
    if format == "xyminwh":
        res_bboxes = []
        for bbox in bboxes:
            cx,cy = bbox[0]+bbox[2]//2,bbox[1]+bbox[3]//2
            new_width = bbox[2]*scale
            new_height = bbox[3]*scale
            min_x = cx-new_width//2
            min_y = cy-new_height//2
            res_bboxes.append((min_x,min_y,new_width,new_height))

        return res_bboxes
    elif format == "yxminmax":
        if not isinstance(bboxes,np.ndarray):
            bboxes = np.array(bboxes)
        ymin = bboxes[...,0]
        xmin = bboxes[...,1]
        ymax = bboxes[...,2]
        xmax = bboxes[...,3]
        h = ymax-ymin
        cy = (ymax+ymin)/2
        w = xmax-xmin
        cx = (xmax+xmin)/2
        nh = h*scale/2
        nw = w*scale/2
        nymin = cy-nh
        nymax = cy+nh
        nxmin = cx-nw
        nxmax = cx+nw

        return np.stack([nymin,nxmin,nymax,nxmax],axis=-1)

'''
bbox:[(xmin,ymin,width,height),....] (format="xyminwh") or [(ymin,xmin,ymax,xmax),...] (format="yxminmax")
size:[H,W]
return a list of new bbox with the size 'size' 
'''
def expand_bbox_by_size(bboxes,size,format="xyminwh"):
    res_bboxes = []
    if format == "xyminwh":
        for bbox in bboxes:
            cx,cy = bbox[0]+bbox[2]//2,bbox[1]+bbox[3]//2
            new_width = size[1]
            new_height = size[0]
            min_x = max(cx-new_width//2,0)
            min_y = max(cy-new_height//2,0)
            res_bboxes.append((min_x,min_y,new_width,new_height))

        return res_bboxes
    elif format == "yxminmax":
        if not isinstance(bboxes,np.ndarray):
            bboxes = np.array(bboxes)
        ymin = bboxes[...,0]
        xmin = bboxes[...,1]
        ymax = bboxes[...,2]
        xmax = bboxes[...,3]
        cy = (ymax + ymin) / 2
        cx = (xmax + xmin) / 2
        nh = size[0]//2
        nw = size[1]//2
        nymin = cy-nh
        nymax = cy+nh
        nxmin = cx-nw
        nxmax = cx+nw

        return np.stack([nymin,nxmin,nymax,nxmax],axis=-1)

'''
bbox:[N,4] (x0,y0,x1,y1)
size:[W,H]

or 

bbox:[N,4] (y0,x0,y1,x1)
size:[H,W]

return a list of new bbox with the minimum size 'size' and same center point
'''
def clamp_bboxes(bboxes,min_size):
    if not isinstance(min_size,Iterable):
        min_size = (min_size,min_size)
    if not isinstance(bboxes,np.ndarray):
        bboxes = np.array(bboxes)
    xmin = bboxes[...,0]
    ymin = bboxes[...,1]
    xmax = bboxes[...,2]
    ymax = bboxes[...,3]
    cy = (ymax + ymin) / 2
    cx = (xmax + xmin) / 2
    h = ymax-ymin
    w = xmax-xmin
    nh = np.maximum(h,min_size[1])/2
    nw = np.maximum(w,min_size[0])/2
    nymin = cy-nh
    nymax = cy+nh
    nxmin = cx-nw
    nxmax = cx+nw

    return np.stack([nxmin,nymin,nxmax,nymax],axis=-1)

'''
bbox:[N,4] (x0,y0,x1,y1)
size:[W,H]

or 

bbox:[N,4] (y0,x0,y1,x1)
size:[H,W]

return a list of new bbox with the minimum size 'size' and same center point
'''
def set_bboxes_size(bboxes,size):
    if not isinstance(size,Iterable):
        size = (size,size)
    if not isinstance(bboxes,np.ndarray):
        bboxes = np.array(bboxes)
    xmin = bboxes[...,0]
    ymin = bboxes[...,1]
    xmax = bboxes[...,2]
    ymax = bboxes[...,3]
    cy = (ymax + ymin) / 2
    cx = (xmax + xmin) / 2
    nh = np.full_like(ymin,size[1]//2)
    nw = np.full_like(xmin,size[0]//2)
    nymin = cy-nh
    nymax = cy+nh
    nxmin = cx-nw
    nxmax = cx+nw

    return np.stack([nxmin,nymin,nxmax,nymax],axis=-1)
'''
bbox:[N,4](x0,y0,x1,y1)
min_size:[W,H]
return a list of new bbox with the minimum size 'size' 
'''
def torch_clamp_bboxes(bboxes,min_size,ignore_zero_size_bboxes=True):
    if not isinstance(min_size,Iterable):
        min_size = (min_size,min_size)
    xmin = bboxes[...,0]
    ymin = bboxes[...,1]
    xmax = bboxes[...,2]
    ymax = bboxes[...,3]
    cy = (ymax + ymin) / 2
    cx = (xmax + xmin) / 2
    h = ymax-ymin
    w = xmax-xmin
    nh = torch.clamp(h,min=min_size[1])/2
    nw = torch.clamp(w,min=min_size[0])/2
    if ignore_zero_size_bboxes:
        nh = torch.where(h>0,nh,h)
        nw = torch.where(w>0,nw,w)
    nymin = cy-nh
    nymax = cy+nh
    nxmin = cx-nw
    nxmax = cx+nw

    return torch.stack([nxmin,nymin,nxmax,nymax],dim=-1)
'''
bboxes:[N,4]
'''
def shrink_box(bboxes,shrink_value=[0,0,0,0]):
    if not isinstance(bboxes,np.ndarray):
        bboxes = np.array(bboxes)
    if not isinstance(shrink_value,list):
        shrink_value = [shrink_value]*4
    ymin,xmin,ymax,xmax = np.transpose(bboxes)
    ymin = ymin+shrink_value[0]
    xmin = xmin+shrink_value[1]
    ymax = ymax-shrink_value[2]
    xmax = xmax-shrink_value[3]
    return np.stack([ymin,xmin,ymax,xmax],axis=1)

'''
boxes:[N,4],[ymin,xmin,ymax,xmax]
return:[ymin,xmin,ymax,xmax]
'''
def bbox_of_boxes(boxes):
    if not isinstance(boxes,np.ndarray):
        boxes = np.array(boxes)
    boxes = np.transpose(boxes)
    ymin = np.min(boxes[0])
    xmin = np.min(boxes[1])
    ymax = np.max(boxes[2])
    xmax = np.max(boxes[3])
    return np.array([ymin,xmin,ymax,xmax])

'''
bbox:[ymin,xmin,ymax,xmax]
'''
def get_bboxes_wh(bboxes):
    return bboxes[...,2:]-bboxes[...,:2]


'''
boxes:[N,4],[ymin,xmin,ymax,xmax]
'''
def absolutely_boxes_to_relative_boxes(boxes,width,height):
    boxes = np.transpose(boxes)
    ymin = boxes[0]/height
    xmin = boxes[1]/width
    ymax = boxes[2]/height
    xmax = boxes[3]/width
    
    return np.stack([ymin,xmin,ymax,xmax],axis=1)

'''
boxes:[N,4],[ymin,xmin,ymax,xmax]
'''
def relative_boxes_to_absolutely_boxes(boxes,width,height):
    boxes = np.transpose(boxes)
    ymin = boxes[0]*(height-1)
    xmin = boxes[1]*(width-1)
    ymax = boxes[2]*(height-1)
    xmax = boxes[3]*(width-1)

    return np.stack([ymin,xmin,ymax,xmax],axis=1)
'''
boxes:[N,4],[ymin,xmin,ymax,xmax]
'''
def relative_boxes_to_absolutely_boxesi(boxes,width,height):
    return relative_boxes_to_absolutely_boxes(boxes=boxes,width=width,height=height).astype(np.int32)

'''
box0:[ymin,xmin,ymax,xmax]
box1:[N,4],[ymin,xmin,ymax,xmax]
用box0裁box1,也就是说box1仅取与box0有重叠的部分
'''
def cut_boxes_by_box0(box0,box1):
    if not isinstance(box1,np.ndarray):
        box1 = np.array(box1)
    ymin,xmin,ymax,xmax = np.transpose(box1)
    ymin = np.minimum(np.maximum(box0[0],ymin),box0[2])
    xmin = np.minimum(np.maximum(box0[1],xmin),box0[3])
    ymax = np.maximum(np.minimum(box0[2],ymax),box0[0])
    xmax = np.maximum(np.minimum(box0[3],xmax),box0[1])
    box1 = np.stack([ymin,xmin,ymax,xmax],axis=0)
    return np.transpose(box1)

def change_bboxes_nr(bboxes0,labels0,bboxes1,labels1,threshold=0.8):
    if not isinstance(labels0,np.ndarray):
        labels0 = np.array(labels0)
    if not isinstance(labels1,np.ndarray):
        labels1 = np.array(labels1)
    nr = labels0.shape[0]
    same_ids = 0
    for i in range(nr):
        box0 = np.array([bboxes0[i]])
        ious = npbboxes_jaccard(box0,bboxes1)
        index = np.argmax(ious)
        if ious[index]>threshold and labels0[i] == labels1[index]:
            same_ids += 1
    return labels0.shape[0]+labels1.shape[0]-2*same_ids


def merge_bboxes(bboxes0,labels0,bboxes1,labels1,iou_threshold=0.5,class_agnostic=True):
    labels1 = np.array(labels1)
    labels0 = np.array(labels0)
    mask = np.ones(labels1.shape,dtype=bool)

    for i in range(labels1.shape[0]):
        if class_agnostic:
            ref_bboxes = bboxes0
        else:
            mask = labels0==labels1[i]
            ref_bboxes = bboxes0[mask]
        if len(ref_bboxes)==0:
            continue

        #ious = npbboxes_jaccard([bboxes1[i]],ref_bboxes)
        ious0 = npbboxes_intersection_of_box0([bboxes1[i]],ref_bboxes)
        ious1 = npbboxes_intersection_of_box0(ref_bboxes,[bboxes1[i]])
        ious = np.concatenate([ious0,ious1],axis=0)

        if np.any(ious>iou_threshold):
            mask[i] = False

    mask = mask.tolist()
    bboxes1 = bboxes1[mask]
    labels1 = labels1[mask]

    bboxes = np.concatenate([bboxes0,bboxes1],axis=0)
    labels = np.concatenate([labels0,labels1],axis=0)

    return bboxes,labels

'''
bboxes0: [N,4]/[1,4] [ymin,xmin,ymax,xmax)
bboxes1: [N,4]/[1,4]ymin,xmin,ymax,xmax)
return:
[-1,1]
'''
def npgiou(bboxes0, bboxes1):
    # 1. calulate intersection over union
    bboxes0 = np.array(bboxes0)
    bboxes1 = np.array(bboxes1)
    area_1 = (bboxes0[..., 2] - bboxes0[..., 0]) * (bboxes0[..., 3] - bboxes0[..., 1])
    area_2 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])

    intersection_wh = np.minimum(bboxes0[..., 2:], bboxes1[..., 2:]) - np.maximum(bboxes0[..., :2], bboxes1[..., :2])
    intersection_wh = np.maximum(intersection_wh, 0)

    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
    union = (area_1 + area_2) - intersection

    ious = intersection / np.maximum(union, 1e-10)

    # 2. (C - (A U B))/C
    C_wh = np.maximum(bboxes0[..., 2:], bboxes1[..., 2:]) - np.minimum(bboxes0[..., :2], bboxes1[..., :2])
    C_wh = np.maximum(C_wh, 1e-10)
    C = C_wh[..., 0] * C_wh[..., 1]

    giou = ious - (C - union) /C
    return giou

'''
bboxes: [N,4] (ymin,xmin,ymax,xmax)
'''
def npbbxoes_nms(bboxes,nms_thrsh=0.5):
    bboxes_nr = len(bboxes)
    bboxes = np.array(bboxes)
    if bboxes_nr<=1:
        return bboxes,[True]
    mask = np.ones([bboxes_nr],dtype=bool)
    for i in range(bboxes_nr-1):
        ious = npbboxes_jaccard([bboxes[i]],bboxes[i+1:])
        for j in range(len(ious)):
            if ious[j]>nms_thrsh:
                mask[i+1+j] = False
    mask = mask.tolist()
    bboxes = bboxes[mask]
    return bboxes,mask

'''
bboxes0: [N,4],(x0,y0,x1,y1)
bboxes1: [M,4],(x0,y0,x1,y1)
return:
[N,M]
'''
def iou_matrix(bboxes0,bboxes1):
    if bboxes0.size==0 or bboxes1.size==0:
        return np.zeros([bboxes0.shape[0],bboxes1.shape[0]],dtype=np.float32)
    bboxes0 = np.array(bboxes0)
    bboxes1 = np.array(bboxes1)
    bboxes0 = np.expand_dims(bboxes0,axis=1)
    bboxes1 = np.expand_dims(bboxes1,axis=0)

    x_int_min = np.maximum(bboxes0[...,0],bboxes1[...,0])
    x_int_max = np.minimum(bboxes0[...,2],bboxes1[...,2])
    y_int_min = np.maximum(bboxes0[...,1],bboxes1[...,1])
    y_int_max = np.minimum(bboxes0[...,3],bboxes1[...,3])

    int_w = np.maximum(x_int_max-x_int_min,0.0)
    int_h = np.maximum(y_int_max-y_int_min,0.0)
    inter_vol = int_w*int_h
    areas0 = np.prod(bboxes0[...,2:]-bboxes0[...,:2],axis=-1)
    areas1 = np.prod(bboxes1[...,2:]-bboxes1[...,:2],axis=-1)
    union_vol = areas0+areas1-inter_vol

    return npsafe_divide(inter_vol,union_vol)

def giou_matrix(atlbrs, btlbrs):
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if ious.size == 0:
        return ious

    for i in range(len(atlbrs)):
        bbox0 = atlbrs[i]
        _gious = npgiou([bbox0], btlbrs)
        ious[i] = _gious
    return ious

def is_point_in_bbox(p,bbox):
    '''

    Args:
        p: (x,y)
        bbox: (x0,y0,x1,y1)

    Returns:
    '''
    if p[0]>=bbox[0] and p[0]<=bbox[2] and p[1]>=bbox[1] and p[1]<=bbox[3]:
        return True
    return False

def is_points_in_bbox(p,bbox):
    '''

    Args:
        p: [N,2] (x,y)
        bbox: (x0,y0,x1,y1)

    Returns:
    '''
    b0 = p[...,0]>=bbox[0] 
    b1 = p[...,0]<=bbox[2] 
    b2 = p[...,1]>=bbox[1] 
    b3 = p[...,1]<=bbox[3]
    b01 = np.logical_and(b0,b1)
    b23 = np.logical_and(b2,b3)
    return np.logical_and(b01,b23)

def make_yolo_target(bboxes,labels):
    '''
    bboxes: list[[N,4]]
    labels: list[[N]]
    '''
    max_nr = max(*[len(x) for x in labels])
    batch_size = len(labels)
    res = np.zeros([batch_size,max_nr, 5], dtype=np.float32)
    if max_nr == 0:
        return res
    for i,bbox in enumerate(bboxes):
        l = labels[i]
        nr = len(l)
        res[i,:nr,0] = l
        res[i,:nr,1:] = bbox
    return res

def trans_mmdet_result(results,labels_trans=None):
    '''
    results: list[np.[X,5]], x0,y0,x1,y1,score, list idx is classes id
    '''
    bboxes = []
    scores = []
    labels = []
    for i in range(len(results)):
        if len(results[i]) == 0:
            continue
        bboxes.append(results[i][:,:4])
        scores.append(results[i][:,4])
        l = i
        if labels_trans is not None:
            l = labels_trans(l)
        l = [l]*results[i].shape[0] 
        labels.extend(l)
    
    if len(bboxes) == 0:
        return np.zeros([0,4],dtype=np.float32),np.zeros([0],dtype=np.int32),np.zeros([0],dtype=np.float32)
    bboxes = np.concatenate(bboxes,axis=0)
    scores = np.concatenate(scores,axis=0)
    labels = np.array(labels)

    return bboxes,labels,scores

def area(bboxes):
    '''
    bboxes: [...,4] (x0,y0,x1,y1) or (y0,x0,y1,x1)
    '''

    s0 = np.maximum(bboxes[...,2]-bboxes[...,0],0)
    s1 = np.maximum(bboxes[...,3]-bboxes[...,1],0)
    return s0*s1


def torch_area(bboxes):
    '''
    bboxes: [...,4] (x0,y0,x1,y1) or (y0,x0,y1,x1)
    '''

    s0 = torch.clamp(bboxes[...,2]-bboxes[...,0],min=0)
    s1 = torch.clamp(bboxes[...,3]-bboxes[...,1],min=0)
    return s0*s1

def correct_bboxes(bboxes,size):
    '''
    bboxes: [N,4]  (x0,y0,x1,y1)
    size: [W,H]
    
    or 
    bboxes: [N,4]  (y0,x0,y1,x1)
    size: [H,W]
    
    '''
    old_type = bboxes.dtype
    bboxes = np.maximum(bboxes,0)
    bboxes = np.minimum(bboxes,np.array([[size[0],size[1],size[0],size[1]]]))
    return bboxes.astype(old_type)

def equal_bboxes(bbox0,bbox1):
    return np.all(bbox0==bbox1)
