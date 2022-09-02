#coding=utf-8
import numpy as np
import cv2
import sys


'''
bboxes:[(ymin,xmin,ymax,xmax),....] value in range[0,1]
mask:[X,h,w]
size:[H,W]
'''
def get_fullsize_mask(boxes,masks,size,mask_bg_value=0):
    dtype = masks.dtype

    res_masks = []
    boxes = np.clip(boxes,0.0,1.0)
    for i,bbox in enumerate(boxes):
        x = int(bbox[1]*size[1])
        y = int(bbox[0]*size[0])
        w = int((bbox[3]-bbox[1])*size[1])
        h = int((bbox[2]-bbox[0])*size[0])
        res_mask = np.ones(size,dtype=dtype)*mask_bg_value
        if w>1 and h>1:
            mask = masks[i]
            mask = cv2.resize(mask,(w,h))
            sys.stdout.flush()
            res_mask[y:y+h,x:x+w] = mask
        res_masks.append(res_mask)

    if len(res_masks)==0:
        return np.zeros([0,size[0],size[1]],dtype=dtype)
    return np.stack(res_masks,axis=0)

def draw_polygon(img,polygon,color=(255,255,255),is_line=True,isClosed=True):
    if is_line:
        return cv2.polylines(img, [polygon], color=color,isClosed=isClosed)
    else:
        return cv2.fillPoly(img,[polygon],color=color)
