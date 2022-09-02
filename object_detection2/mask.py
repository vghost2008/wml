#coding=utf-8
import numpy as np

'''
mask: [N,H,W] value is 0 or 1
labels: [N] labels of mask
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
            pos_mask = mask[i].astype(np.bool)
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
                pos_mask = mask[i].astype(np.bool)
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
            pos_mask = mask[i].astype(np.bool)
            res_mask[pos_mask] = labels[i]
        return res_mask
