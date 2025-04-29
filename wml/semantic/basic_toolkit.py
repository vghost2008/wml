import numpy as np
import cv2
import wml.basic_img_utils as bwmli
import sys
from ..threadtoolkit import par_for_each,DEFAULT_THREAD_NR
from functools import partial

def find_contours_in_bbox(mask,bbox):
    bbox = np.array(bbox).astype(np.int32)
    sub_mask = bwmli.crop_img_absolute_xy(mask,bbox)
    if sub_mask.shape[0]<=1 or sub_mask.shape[1]<=1:
        return []
    contours,hierarchy = cv2.findContours(sub_mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return []
    offset = np.array([bbox[0],bbox[1]],dtype=np.int32)
    offset = np.reshape(offset,[1,1,2])
    res = []
    for x in contours:
        res.append(x+offset)
    return res

def get_bboxes_by_contours(contours):
    '''
    contours:[[N,2]]
    '''
    if len(contours)==0:
        return np.zeros([4],dtype=np.float32)
    cn0 = np.reshape(contours[0],[-1,2])
    x0 = np.min(cn0[:,0])
    x1 = np.max(cn0[:,0])
    y0 = np.min(cn0[:,1])
    y1 = np.max(cn0[:,1])
    for cn in contours[1:]:
        cn = np.reshape(cn,[-1,2])
        x0 = min(np.min(cn[:,0]),x0)
        x1 = max(np.max(cn[:,0]),x1)
        y0 = min(np.min(cn[:,1]),y0)
        y1 = max(np.max(cn[:,1]),y1)

    return np.array([x0,y0,x1,y1],dtype=np.float32)

def findContours(mask,mode=cv2.RETR_TREE,method=cv2.CHAIN_APPROX_SIMPLE):
    '''
    mask: [H,W] value is 0 or 1, np.uint8
    return:
    contours: list[[N,2]] 
    '''

    _contours, hierarchy = cv2.findContours(mask, mode,method)
    try:
        hierarchy = np.reshape(hierarchy,[-1,4]) 
    except Exception as e:
        if len(_contours) != 0:
            print(f"ERROR: {e}, {_contours} {hierarchy}")
        hierarchy = np.zeros([0,4])
        sys.stdout.flush()
    contours = []
    for he,cont in zip(hierarchy,_contours):
        if he[-1]>=0 and cv2.contourArea(cont) < cv2.contourArea(_contours[he[-1]]):
            continue
        if len(cont.shape) == 3 and cont.shape[1] == 1:
            contours.append(np.squeeze(cont,axis=1))
        elif len(cont.shape)==2 and cont.shape[0]>2:
            contours.append(cont)
    return contours,hierarchy

def npresize_mask(mask,size=None,scale_factor=None):
    '''
    mask: [N,H,W]
    size: (new_w,new_h)
    '''
    if scale_factor is not None and size is None:
        size = (int(mask[0].shape[1]*scale_factor),int(mask[0].shape[0]*scale_factor))
    if len(mask) == 0:
        return np.zeros([0,size[1],size[0]],dtype=mask.dtype)
    new_mask = []
    for i in range(len(mask)):
        cur_m = cv2.resize(mask[i],dsize=(size[0],size[1]),interpolation=cv2.INTER_NEAREST)
        new_mask.append(cur_m)
    new_mask = np.stack(new_mask,axis=0)
    return new_mask

def npresize_mask_mt(mask,size=None,scale_factor=None,thread_nr=0):
    '''
    mask: [N,H,W]
    size: (new_w,new_h)
    '''
    if scale_factor is not None and size is None:
        size = (int(mask[0].shape[1]*scale_factor),int(mask[0].shape[0]*scale_factor))
    if len(mask) == 0:
        return np.zeros([0,size[1],size[0]],dtype=mask.dtype)

    if thread_nr <= 0:
        thread_nr = min(16,DEFAULT_THREAD_NR)
    
    fn = partial(cv2.resize,dsize=(size[0],size[1]),interpolation=cv2.INTER_NEAREST)
    new_mask = par_for_each(mask,fn,thread_nr=thread_nr)
    new_mask = np.stack(new_mask,axis=0)
    return new_mask