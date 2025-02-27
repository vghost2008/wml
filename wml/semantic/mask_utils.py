#coding=utf-8
import numpy as np
import logging
import wml.basic_img_utils as bwmli
import cv2
from .basic_toolkit import *
import torch
import math

def np_iou(mask0,mask1):
    if mask0.dtype is not bool:
        mask0 = mask0.astype(bool)
    if mask1.dtype is not bool:
        mask1 = mask1.astype(bool)

    if len(mask0.shape) != len(mask1.shape):
        logging.warning("Mask not compatible with each other")
        return 0.

    different = np.logical_xor(mask0,mask1)
    different = different.astype(np.float32)
    different = np.sum(different)

    union = np.logical_or(mask0,mask1)
    union = union.astype(np.float32)
    union = np.sum(union)
    #logging.info("union={}, different={}, mask={}, gt={}".format(union,different,np.sum(mask0.astype(np.float32)),np.sum(mask1.astype(np.float32))))

    if union == 0:
        return 100.0

    return 100.0-different*100.0/union

'''
mask:[H,W,NUM_CLASSES]
mask:[H,W]
'''
def np_mask2masklabels(mask,begin_label=1):
    res = np.zeros(mask.shape[:2],np.int32)
    h = mask.shape[0]
    w = mask.shape[1]
    num_classes = mask.shape[2]

    for i in range(h):
        for j in range(w):
            for k in range(num_classes):
                if mask[i,j,k]>0:
                    res[i,j] = k+begin_label
                    break

    return res

def resize_img_and_mask(img,mask,size,img_pad_value=127,mask_pad_value=255,pad_type=1):
    '''

    Args:
        img:
        mask:
        size: (w,h)
        img_pad_value:
        mask_pad_value:
        pad_type:

    Returns:

    '''
    img = np.array(img)
    mask = np.array(mask)
    img = bwmli.resize_img(img, size, keep_aspect_ratio=True)
    mask = bwmli.resize_img(mask, img.shape[:2][::-1], keep_aspect_ratio=False, interpolation=cv2.INTER_NEAREST)
    img, px0, px1, py0, py1 = bwmli.pad_img(img, size, pad_value=img_pad_value, pad_type=pad_type, return_pad_value=True)
    mask = bwmli.pad_imgv2(mask, px0, px1, py0, py1, pad_value=mask_pad_value)
    return img,mask

'''
mask:[H,W] value is 1 or 0
rect:[ymin,xmin,ymax,xmax]
output:
the new mask in sub image and correspond bbox
'''
def cut_mask(mask,rect):
    max_area = np.sum(mask)
    cuted_mask = bwmli.sub_image(mask,rect)
    ratio = np.sum(cuted_mask)/max(1,max_area)
    if ratio <= 1e-6:
        return None,None,ratio
    ys,xs = np.where(cuted_mask)
    xmin = np.min(xs)
    xmax = np.max(xs)
    ymin = np.min(ys)
    ymax = np.max(ys)
    bbox = np.array([xmin,ymin,xmax,ymax],dtype=np.int32)
    return cuted_mask,bbox,ratio

'''
mask:[N,H,W] value is 1 or 0
bboxes:[N,4] [ymin,xmin,ymax,xmax]
output:
the new mask in sub image and correspond bbox
'''
def cut_masks(masks,bboxes):
    new_masks = []
    new_bboxes = []
    ratios = []
    nr = len(masks)
    for i in range(len(nr)):
        n_mask,n_bbox,ratio = cut_mask(masks[i],bboxes[i])
        new_masks.append(n_mask)
        new_bboxes.append(n_bbox)
        ratios.append(ratio)
    
    return new_masks,new_bboxes,ratios


def resize_mask(mask,size=None,r=None,mode='nearest'):
    '''
    mask: [N,H,W]
    size: (new_w,new_h)
    mode (str): algorithm used for upsampling: 'nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area' | 'nearest-exact'.
                Default: 'nearest'
    '''
    if size is None:
        size = (int(mask.shape[2]*r),int(mask.shape[1]*r))
    if mask.numel()==0:
        return mask.new_zeros([mask.shape[0],size[1],size[0]])

    mask = torch.unsqueeze(mask,dim=0)
    mask =  torch.nn.functional.interpolate(mask,size=(size[1],size[0]),mode=mode)
    mask = torch.squeeze(mask,dim=0)
    return mask

def npresize_mask(mask,size=None,r=None):
    '''
    mask: [N,H,W]
    size: (new_w,new_h)
    '''
    if mask.shape[0]==0:
        return np.zeros([0,size[1],size[0]],dtype=mask.dtype)
    new_mask = []
    for i in range(mask.shape[0]):
        cur_m = cv2.resize(mask[i],dsize=(size[0],size[1]),interpolation=cv2.INTER_NEAREST)
        new_mask.append(cur_m)
    new_mask = np.stack(new_mask,axis=0)
    return new_mask

def resize_mask_structures(mask,size):
    '''
    size:[W,H]
    '''
    if isinstance(mask,np.ndarray):
        return npresize_mask(mask,size)
    if torch.is_tensor(mask):
        return resize_mask(mask,size)
    if hasattr(mask,"resize"):
        return mask.resize(size)
    raise RuntimeError("Unimplement")
