#coding=utf-8
import numpy as np
import wml.object_detection2.bboxes as odb
import wml.img_utils as wmli
import copy
from .mask import get_bboxes_by_mask
from itertools import count


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

class WCrop:
    '''
    '''

    def __init__(self,
                 img_pad_value=127,
                 mask_pad_value=0,
                 bbox_keep_ratio=0.2,
                 mask_domina=True):
                 
        self.img_pad_value = img_pad_value
        self.mask_pad_value = mask_pad_value
        self.bbox_keep_ratio = bbox_keep_ratio
        self.mask_domina = mask_domina

    def apply(self, results,crop_bbox):
        """Random crop and around padding the original image.

        Args:
            results (dict): Image infomations in the augment pipeline.

        Returns:
            results (dict): The updated dict.
        """
        img = results['img']
        patch = crop_bbox
        try:
            cropped_img = wmli.crop_and_pad(img, patch,pad_color=self.img_pad_value)
        except:
            print("Crop error:",patch)

        x_offset = patch[0]
        y_offset = patch[1]
        new_w = patch[2]-x_offset
        new_h = patch[3]-y_offset
        results['img'] = cropped_img
        results['img_shape'] = cropped_img.shape
        results['pad_shape'] = cropped_img.shape

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', ['gt_bboxes']):
            bboxes = results[key]
            old_bboxes = copy.deepcopy(bboxes)
            old_area = odb.area(old_bboxes)
            bboxes[:, 0:4:2] -= x_offset
            bboxes[:, 1:4:2] -= y_offset
            bboxes[:, 0:4:2] = np.clip(bboxes[:, 0:4:2], 0, new_w)
            bboxes[:, 1:4:2] = np.clip(bboxes[:, 1:4:2], 0, new_h)
            keep0 = (bboxes[:, 2] > bboxes[:, 0]) & (
                bboxes[:, 3] > bboxes[:, 1])
            new_area = odb.area(bboxes)
            area_ratio = new_area/(old_area+1e-6)
            keep1 = area_ratio>self.bbox_keep_ratio
            keep = np.logical_and(keep0,keep1)
            bboxes = bboxes[keep]
            results[key] = bboxes
            if key in ['gt_bboxes']:
                if 'gt_labels' in results:
                    labels = results['gt_labels']
                    labels = labels[keep]
                    results['gt_labels'] = labels
                if 'gt_masks' in results:
                    gt_masks = results['gt_masks']
                    gt_masks = gt_masks[keep]
                    gt_masks = wmli.crop_masks_absolute_xy(gt_masks,patch)
                    results['gt_masks'] = gt_masks

                    if self.mask_domina:
                        old_area = old_area[keep]
                        bboxes = get_bboxes_by_mask(gt_masks)
                        new_area = odb.area(bboxes)
                        area_ratio = new_area/(old_area+1e-6)
                        keep = area_ratio>self.bbox_keep_ratio
                        bboxes = bboxes[keep]
                        results[key] = bboxes
                        if 'gt_labels' in results:
                            labels = results['gt_labels']
                            labels = labels[keep]
                            results['gt_labels'] = labels
                        gt_masks = gt_masks[keep]
                        results['gt_masks'] = gt_masks

            return results

    def __call__(self, results,crop_bbox):
        return self.apply(results,crop_bbox)


def make_text2label(classes=None,label_text2id={}):
    res = {}
    if classes is not None:
        tmp_d = dict(zip(classes,count()))
        res.update(tmp_d)
    
    if label_text2id is not None:
        for k,v in label_text2id.items():
            if isinstance(v,(str,bytes)) and v in res:
                res[k] = res[v]
            else:
                res[k] = v
    
    for k,v in list(res.items()):
        if isinstance(v,(str,bytes)) and v in res:
            res[k] = res[v]
    
    return res


