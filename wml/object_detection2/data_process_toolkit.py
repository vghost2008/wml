import wml.img_utils as wmli
import wml.object_detection2.bboxes as odb
import cv2 as cv
import numpy as np
from wml.semantic.mask_utils import cut_mask

def motion_blur(image, degree=10, angle=20):
    image = np.array(image)
    # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
    M = cv.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv.warpAffine(motion_blur_kernel, M, (degree, degree))

    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv.filter2D(image, -1, motion_blur_kernel)
    # convert to uint8
    cv.normalize(blurred, blurred, 0, 255, cv.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred

'''
cnt is a contour in a image, cut the area of rect
cnt:[[x,y],[x,y],...]
rect:[ymin,xmin,ymax,xmax]
output:
the new contours in sub image
'''
def cut_contour(cnt,rect):
    bbox = bbox_of_contour(cnt)
    width = max(bbox[3],rect[3])
    height = max(bbox[2],rect[2])
    img = np.zeros(shape=(height,width),dtype=np.uint8)
    segmentation = cv.drawContours(img,[cnt],-1,color=(1),thickness=cv.FILLED)
    cuted_img = wmli.sub_image(segmentation,rect)
    contours,hierarchy = cv.findContours(cuted_img,cv.CV_RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
    return contours

'''
find the contours in rect of segmentation
segmentation:[H,W] value is 1 or 0
rect:[ymin,xmin,ymax,xmax]
output:
the new contours in sub image and correspond bbox
'''
def cut_contourv2(segmentation,rect):
    org_contours,org_hierarchy = cv.findContours(segmentation,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
    max_area = np.sum(segmentation)
    cuted_img = wmli.sub_image(segmentation,rect)
    contours,hierarchy = cv.findContours(cuted_img,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
    boxes = []
    ratio = []
    for cnt in contours:
        boxes.append(bbox_of_contour(cnt))
        ratio.append(cv.contourArea(cnt)/max_area)
    return contours,boxes,ratio

def remove_class_in_image(bboxes,labels,labels_to_remove,image,default_value=127,scale=1.1):
    bboxes = bboxes.astype(np.int32)
    mask = np.ones_like(labels,dtype=bool)
    for l in labels_to_remove:
        tm = labels==l
        mask = np.logical_and(tm,mask)
    keep_mask = np.logical_not(mask)
    keep_bboxes = bboxes[keep_mask]
    remove_bboxes = bboxes[mask]
    img_mask = np.ones(image.shape[:2],dtype=bool)

    wmli.remove_boxes_of_img(img_mask,remove_bboxes,False)
    if scale>1.0:
        t_keep_bboxes = npscale_bboxes(keep_bboxes,scale).astype(np.int32)
    else:
        t_keep_bboxes = keep_bboxes
    wmli.remove_boxes_of_img(img_mask,t_keep_bboxes,True)

    img_mask = np.expand_dims(img_mask,axis=-1)
    img_mask = np.tile(img_mask,[1,1,3])
    remove_image = np.ones_like(image)*default_value
    image = np.where(img_mask,image,remove_image)
    return image,keep_bboxes,labels[keep_mask]

def remove_class(bboxes,labels,scores=None,labels_to_remove=[]):
    bboxes = bboxes.astype(np.int32)
    mask = np.ones_like(labels,dtype=bool)
    for i,l in enumerate(labels):
        if l in labels_to_remove:
            mask[i] = False
    bboxes = bboxes[mask]
    labels = labels[mask]
    if scores is not None:
        scores = scores[mask]
    
    return bboxes,labels,scores,mask


def cut_annotation(cut_bbox,img,labels,bboxes,masks=None,adjust_bbox=True,keep_ratio=0):
    '''
    cut_bbox: [y0,x0,y1,x1]
    img: [H,W,3/1]
    bboxes: [N,4] [y0,x0,y1,x1]
    masks: [N,H,W]
    '''
    cut_bbox = list(cut_bbox)
    if adjust_bbox:
        b_w = cut_bbox[3]-cut_bbox[1]
        b_h = cut_bbox[2]-cut_bbox[0]
        cut_bbox[0] = min(img.shape[0],cut_bbox[2])-b_h
        cut_bbox[1] = min(img.shape[1],cut_bbox[3])-b_w

    cut_bbox[0] = max(0,cut_bbox[0])
    cut_bbox[1] = max(0,cut_bbox[1])

    cut_bbox[2] = min(cut_bbox[2],img.shape[0])
    cut_bbox[3] = min(cut_bbox[3],img.shape[1])

    new_bboxes = []
    new_labels = []
    new_img = wmli.sub_image(img,cut_bbox)
    if masks is not None:
        new_masks = []
        for i in range(len(labels)):
            n_mask,n_bbox,ratio = cut_mask(masks[i],cut_bbox)
            if n_mask is not None and n_bbox is not None and ratio>keep_ratio:
                new_bboxes.append(n_bbox)
                new_labels.append(labels[i])
                new_masks.append(n_mask)
        if len(new_labels)>0:
            new_labels = np.concatenate(new_labels,axis=0)
            new_bboxes = np.concatenate(new_bboxes,axis=0)
            new_masks = np.concatenate(new_masks,axis=0)
        else:
            new_labels = np.zeros([0],dtype=labels.dtype)
            new_bboxes = np.zeros([0,4],dtype=bboxes.dtype)
            new_masks = np.zeros([0,new_img.shape[0],new_img.shape[1]],dtype=masks.dtype)
    else:
        new_masks = None
        bbox_area = odb.area(bboxes)
        new_bboxes = odb.cut_boxes_by_box0(box0=cut_bbox,box1=bboxes)
        new_bbox_area = odb.area(new_bboxes)
        ratios = new_bbox_area/np.maximum(bbox_area,1)
        keep = ratios>keep_ratio
        new_labels = labels[keep]
        new_bboxes = new_bboxes[keep]
        offset = np.reshape(np.array([cut_bbox[0],cut_bbox[1],cut_bbox[0],cut_bbox[1]],dtype=new_bboxes.dtype),[1,4])
        new_bboxes = new_bboxes-offset

    return new_img,new_labels,new_bboxes,new_masks
