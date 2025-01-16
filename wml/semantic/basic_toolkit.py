import numpy as np
import cv2
import wml.basic_img_utils as bwmli
import sys

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

