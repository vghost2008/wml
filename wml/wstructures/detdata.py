from collections import namedtuple
import copy
import numpy as np

DetData = namedtuple('DetData','path,img_shape,labels,labels_name,bboxes,masks,area,is_crowd,extra_data') #img_shape:(H,W), masks 为全图np.ndarray或WPolygonMasks
DetBboxesData = namedtuple('DetBboxesdata','path,img_shape,labels,bboxes,is_crowd')

def detbboxesdata2detdata(bboxes_data):
    return DetData(bboxes_data.path,bboxes_data.img_shape,bboxes_data.labels,None,bboxes_data.bboxes,None,None,bboxes_data.is_crowd,None)

def detdata2detbboxesdata(bboxes_data):
    return DetBboxesData(bboxes_data.path,bboxes_data.img_shape,bboxes_data.labels,bboxes_data.bboxes,bboxes_data.is_crowd)

def concat(datas):
    if len(datas) == 0:
        return datas
    if len(datas) == 1:
        return datas[0]
    
    if isinstance(datas[0],DetBboxesData):
        return concat_detbboxesdata(datas)
    
    return concat_detdata(datas)


def concat_detdata(datas):
    if len(datas) == 0:
        return datas
    if len(datas) == 1:
        return datas[0]
    res = copy.deepcopy(datas[0])

    labels = []
    labels_name = [] if res.labels_name is not None else None
    bboxes = []
    masks = [] if res.masks is not None else None
    is_crowd = [] if res.is_crowd is not None else None
    for d in datas:
        labels.append(np.array(d.labels))
        if labels_name is not None:
            labels_name.append(np.array(d.labels_name,dtype=object))
        bboxes.append(d.bboxes)
        if masks is not None:
            masks.append(d.masks)
        if is_crowd is not None:
            is_crowd.append(d.is_crowd)
    
    labels = np.concatenate(labels,axis=0)
    if labels_name is not None:
        labels_name = np.concatenate(labels_name,axis=0)
    
    bboxes = np.concatenate(bboxes,axis=0)
    if masks is not None:
        if isinstance(masks[0],np.ndarray):
            masks = np.concatenate(masks,axis=0)
        if isinstance(masks[0],(list,tuple)):
            _masks = []
            for mask in masks:
                _masks.extend(list(mask))
            masks = _masks
        else:
            masks = type(masks[0]).concatenate(masks)
    
    if is_crowd is not None:
        is_crowd = np.concatenate(is_crowd,axis=0)

    return DetData(res.path,res.img_shape,labels,labels_name,bboxes,masks,None,is_crowd,res.extra_data)

def concat_detbboxesdata(datas):
    datas = [detbboxesdata2detdata(x) for x in datas]
    data = concat_detdata(datas)
    return detdata2detbboxesdata(data)