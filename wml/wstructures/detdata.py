from collections import namedtuple
import copy
import numpy as np

'''
bbox: [N,4], y0,x0,y1,x1, absolute_coord
'''
#path: img path, img_shape:(H,W), masks 为全图np.ndarray[N,H,W]或WPolygonMasks
DetData = namedtuple('DetData','path,img_shape,labels,labels_name,bboxes,masks,area,is_crowd,extra_data') 
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


def detdata_remove_label_names(detdata,names2remove):
    if not isinstance(names2remove,(list,tuple)):
        names2remove = [names2remove]

    keep = []
    for l in detdata.labels_name:
        if l in names2remove:
            keep.append(False)
        else:
            keep.append(True)
    keep = np.array(keep,dtype=bool)
    labels = detdata.labels
    if labels is not None and len(labels)>0:
        labels = labels[keep]
    labels_name = detdata.labels_name[keep]
    bboxes = detdata.bboxes[keep]
    masks = detdata.masks
    if masks is not None and len(masks)>0:
        masks = masks[keep]
    area = detdata.area
    if area is not None and len(area)>0:
        area = area[keep]
    is_crowd = detdata.is_crowd
    if is_crowd is not None and len(is_crowd)>0:
        is_crowd = is_crowd[keep]
    detdata = detdata._replace(labels=labels,labels_name=labels_name,bboxes=bboxes,masks=masks,area=area,is_crowd=is_crowd)

    return detdata

def detbboxes_remove_label_names(detdata,names2remove):
    if not isinstance(names2remove,(list,tuple)):
        names2remove = [names2remove]

    keep = []
    for l in detdata.labels:
        if l.strip() in names2remove:
            keep.append(False)
        else:
            keep.append(True)
    keep = np.array(keep,dtype=bool)
    labels = np.array(detdata.labels)
    if labels is not None and len(labels)>0:
        labels = labels[keep]
    bboxes = detdata.bboxes[keep]
    is_crowd = detdata.is_crowd
    if is_crowd is not None and len(is_crowd)>0:
        is_crowd = np.array(is_crowd)[keep]
    detdata = detdata._replace(labels=labels,bboxes=bboxes,is_crowd=is_crowd)

    return detdata

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


def resize_detdata(data,size,old_size=None):
    '''
    size: [w,h]
    '''

    if isinstance(data,DetBboxesData):
        return resize_detbboxesdata(data,size,old_size)

    img_shape = data.img_shape[:2] #H,W

    masks = data.masks
    if masks is not None:
        masks = masks.resize(size)

    bboxes = data.bboxes
    if bboxes is not None:
        w_scale = size[0]/img_shape[1]
        h_scale = size[1]/img_shape[0]
        old_dtype = bboxes.dtype
        bboxes = (bboxes*np.array([[w_scale,h_scale,w_scale,h_scale]])).astype(old_dtype)

    img_shape = [size[1],size[0]]
    data = data._replace(img_shape=img_shape,masks=masks,bboxes=bboxes)

    return data

def resize_detbboxesdata(data,size,old_size=None):
    '''
    size: [w,h]
    old_size: [w,h]
    '''
    bboxes = data.bboxes
    if bboxes is not None:
        w_scale = size[0]/old_size[0]
        h_scale = size[1]/old_size[1]
        old_dtype = bboxes.dtype
        bboxes = (bboxes*np.array([[w_scale,h_scale,w_scale,h_scale]])).astype(old_dtype)

    img_shape = [size[1],size[0]]
    data = data._replace(img_shape=img_shape,bboxes=bboxes)

    return data