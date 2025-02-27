#coding=utf-8
import numpy as np

'''
mask: [N,H,W] value is 0 or 1
labels: [N] labels of mask
return:
[H,W] value is labels' value
'''
def dense_mask_to_sparse_mask(mask:np.ndarray,labels,default_label=0):
    if len(labels) == 0 and not isinstance(mask,np.ndarray):
        return None
    elif len(labels)==0:
        _,H,W = mask.shape
        return np.ones([H,W],dtype=np.int32)*default_label
    else:
        N,H,W = mask.shape
        res_mask = np.ones([H,W],dtype=np.int32)*default_label
        for i in range(N):
            pos_mask = mask[i].astype(bool)
            res_mask[pos_mask] = labels[i]
        return res_mask

def dense_mask_to_sparse_maskv2(mask:np.ndarray,labels,labels_order,default_label=0):
    '''
    generate the mask by labels order
    Args:
        mask:
        labels:
        labels_order:
        default_label:

    Returns:

    '''
    labels = list(labels)
    if len(labels) == 0 and not isinstance(mask,np.ndarray):
        return None
    elif len(labels)==0:
        _,H,W = mask.shape
        return np.ones([H,W],dtype=np.int32)*default_label
    else:
        N,H,W = mask.shape
        res_mask = np.ones([H,W],dtype=np.int32)*default_label
        for l in labels_order:
            if l not in labels:
                continue
            for i in range(N):
                if labels[i] != l:
                    continue
                pos_mask = mask[i].astype(bool)
                res_mask[pos_mask] = l
        return res_mask


'''
mask: [N,H,W] value is 0 or 1
labels: [N] labels of mask
small object overwrite big object 
'''
def dense_mask_to_sparse_maskv3(mask:np.ndarray,labels,default_label=0):
    if len(labels) == 0 and not isinstance(mask,np.ndarray):
        return None
    elif len(labels)==0:
        _,H,W = mask.shape
        return np.ones([H,W],dtype=np.int32)*default_label
    else:
        N,H,W = mask.shape
        nrs = np.sum(mask,axis=(1,2))
        orders = np.argsort(-nrs)
        res_mask = np.ones([H,W],dtype=np.int32)*default_label
        for i in orders:
            pos_mask = mask[i].astype(bool)
            res_mask[pos_mask] = labels[i]
        return res_mask

def get_bboxes_by_mask(masks):
    '''
    masks: [N,H,W]
    return :
    [N,4] (xmin,ymin,xmax,ymax)
    '''
    if len(masks) == 0:
        return np.zeros([0,4],dtype=np.float32)

    bboxes = []
    for i in range(masks.shape[0]):
        cur_mask = masks[i]
        idx = np.nonzero(cur_mask)
        xs = idx[1]
        ys = idx[0]
        if len(xs)==0:
            bboxes.append(np.zeros([4],dtype=np.float32))
        else:
            x0 = np.min(xs)
            y0 = np.min(ys)
            x1 = np.max(xs)
            y1 = np.max(ys)
            bboxes.append(np.array([x0,y0,x1,y1],dtype=np.float32))
    
    bboxes = np.array(bboxes)

    return bboxes

def crop_masks_by_bboxes(masks,bboxes):
    '''
    masks: [N,H,W]
    bboxes: [N,4] (x0,y0,x1,y1), absolute coordinate
    '''
    shape = masks.shape[1:]
    bboxes[:,0:4:2] = np.clip(bboxes[:,0:4:2],0.0,shape[1])
    bboxes[:,1:4:2] = np.clip(bboxes[:,1:4:2],0.0,shape[0])
    bboxes = bboxes.astype(np.int32)
    res = []
    for i in range(bboxes.shape[0]):
        x0,y0,x1,y1 = bboxes[i]
        m = masks[i,y0:y1,x0:x1].copy()
        res.append(m)
    
    return res
