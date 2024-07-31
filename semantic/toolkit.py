#coding=utf-8
import sys
import os
from .mask_utils import np_iou
import numpy as np
from .visualization_utils import MIN_RANDOM_STANDARD_COLORS, draw_mask_on_image_array

'''
image:[height,width,3]
mask:[height,width,N]
colors:[N], string
alpha:
'''
def np_draw_masks_on_image(image, mask, colors, alpha=0.4):
    if image.dtype is not np.uint8:
        image = image.astype(np.uint8)
    if mask.dtype is not np.uint8:
        mask = mask.astype(np.uint8)
    mask = np.transpose(mask, axes=[2, 0, 1])
    colors_nr = len(colors)

    for i,msk in enumerate(mask):
        image = draw_mask_on_image_array(image,msk,colors[i%colors_nr],alpha)

    return image

'''
image:[batch_size,height,width,3]
mask:[batch_size,height,width,N]
alpha:
'''
def np_draw_masks_on_images(image,mask,alpha,colors=MIN_RANDOM_STANDARD_COLORS,no_first_mask=False):
    if no_first_mask:
        mask = mask[:,:,:,1:]
    res_images = []

    for img,msk in zip(image,mask):
        new_img = np_draw_masks_on_image(image=img,mask=msk,colors=colors,alpha=alpha)
        res_images.append(new_img)

    return np.array(res_images)



'''
masks:[X,H,W]
labels:[X]
no_background: 如果为True, 那么labels的值域为[1,num_classes], 生成时labels转换为labels-1
output:
[num_classes,H,W]/[num_classes-1,H,W](no_background=True)
'''
def merge_masks(masks,labels,num_classes,size=None,no_background=False):
    if size is not None:
        width = size[1]
        height = size[0]
    elif len(masks.shape)>=3:
        width = masks.shape[2]
        height = masks.shape[1]

    if no_background:
        get_label = lambda x:max(0,x-1)
        res = np.zeros([num_classes-1,height,width],dtype=np.int32)
    else:
        get_label = lambda x:x
        res = np.zeros([num_classes,height,width],dtype=np.int32)

    for i,mask in enumerate(masks):
        label = get_label(labels[i])
        res[label:label+1,:,:] = np.logical_or(res[label:label+1,:,:],np.expand_dims(mask,axis=0))

    return res

'''
def get_fullsize_merged_mask(masks,bboxes,labels,size,num_classes,no_background=True):
    fullsize_masks = ivs.get_fullsize_mask(bboxes,masks,size)
    return merge_masks(fullsize_masks,labels,num_classes,size,no_background)
'''

class ModelPerformance:
    def __init__(self,no_first_class=True):
        self.test_nr = 0
        self.total_iou = 0.
        self.no_first_class = no_first_class


    def clear(self):
        self.test_nr = 0
        self.total_iou = 0.

    '''
    mask_gt: [batch_size,h,w,num_classes]
    mask_pred: [batch_size,h,w,num_classes]
    background is [:,:,0]
    '''
    def __call__(self, mask_gt,mask_pred):
        if self.no_first_class:
            mask_gt = mask_gt[:,:,:,1:]
            mask_pred = mask_pred[:,:,:,1:]
        tmp_iou = np_iou(mask_gt,mask_pred)
        self.total_iou += tmp_iou
        self.test_nr += 1
        return tmp_iou, self.mIOU()

    def mIOU(self):
        return self.total_iou/self.test_nr
