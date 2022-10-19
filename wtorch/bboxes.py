import torch
import torch.nn.functional as F

def cxywh2xy(bboxes):
    cxy = bboxes[...,:2]
    hwh = bboxes[...,2:]/2
    minxy = cxy-hwh
    maxxy = cxy+hwh
    return torch.cat([minxy,maxxy],dim=-1)

def xy2cxywh(bboxes):
    wh = bboxes[...,2:]-bboxes[...,:2]
    cxy = (bboxes[...,2:]+bboxes[...,:2])/2
    return torch.cat([cxy,wh],dim=-1)

def distored_boxes(bboxes:torch.Tensor,scale=[0.8,1.2],offset=0.2):
    bboxes = xy2cxywh(bboxes)
    cxy,wh = torch.split(bboxes,2,dim=-1)
    wh_scales = torch.rand(list(wh.shape),dtype=bboxes.dtype)*(scale[1]-scale[0])+scale[0]
    wh_scales = wh_scales.to(wh.device)
    wh = wh*wh_scales
    cxy_offset = torch.rand(list(cxy.shape),dtype=cxy.dtype)*offset
    cxy_offset = cxy_offset.to(cxy.device)
    cxy = cxy+cxy_offset
    bboxes = torch.cat([cxy,wh],axis=-1)
    bboxes = cxywh2xy(bboxes)
    bboxes = torch.nn.functional.relu(bboxes)
    return bboxes

def flip(bboxes,size):
    '''
    bboxes:[N,4][xmin,ymin,xmax,ymax]
    size:[H,W]
    '''
    _bboxes = torch.clone(bboxes)
    _bboxes[...,0] = size[1]-bboxes[...,2]
    _bboxes[...,2] = size[1]-bboxes[...,0]
    return _bboxes

def bboxes_ious(bboxesa, bboxesb):
    '''
    bboxesa: [N,4] or [1,4] (xmin,ymin,xmax,ymax)
    bboxesb: [N,4] or [1,4] (xmin,ymin,xmax,ymax)
    return:
    [N]
    '''

    bboxesa = torch.unbind(bboxesa,-1)
    bboxesb = torch.unbind(bboxesb,-1)
    int_xmin = torch.maximum(bboxesa[0], bboxesb[0])
    int_ymin = torch.maximum(bboxesa[1], bboxesb[1])
    int_xmax = torch.minimum(bboxesa[2], bboxesb[2])
    int_ymax = torch.minimum(bboxesa[3], bboxesb[3])
    h = F.relu(int_ymax - int_ymin)
    w = F.relu(int_xmax - int_xmin)
    inter_vol = h * w
    union_vol = -inter_vol \
                + (bboxesa[2] - bboxesa[0]) * (bboxesa[3] - bboxesa[1]) \
                + (bboxesb[2] - bboxesb[0]) * (bboxesb[3] - bboxesb[1])
    min_value = torch.full_like(union_vol,1e-8)
    jaccard = inter_vol/torch.maximum(union_vol, min_value)
    return jaccard

def bboxes_ious_matrix(bboxesa,bboxesb):
    '''
    bboxesa: [N,4]  (xmin,ymin,xmax,ymax)
    bboxesb: [M,4]  (xmin,ymin,xmax,ymax)
    return:
    [N,M]
    '''
    ious = []
    for bbox in bboxesa:
        _ious = bboxes_ious(bbox,bboxesb)
        ious.append(_ious)
    return torch.stack(ious,dim=0)



