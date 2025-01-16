import wml.wml_utils as wmlu
import os
import json
import numpy as np
import cv2 as cv
import copy
import wml.img_utils as wmli
import random
import matplotlib.pyplot as plt
import sys
import cv2
from wml.object_detection2.standard_names import *
import wml.object_detection2.bboxes as odb
from functools import partial
from .common import resample
from wml.semantic.structures import *
from wml.semantic.basic_toolkit import findContours
import glob
import math
from wml.walgorithm import points_on_circle
import copy




def trans_odresult_to_annotations_list(data):
    labels = data[RD_LABELS]
    res = []
    for i in range(len(labels)):
        annotation = {}
        annotation["category_id"] = labels[i]
        annotation["segmentation"] = data[RD_FULL_SIZE_MASKS]
        res.append(annotation)

    return res

def trans_absolute_coord_to_relative_coord(image_info,annotations_list):
    H = image_info['height']
    W = image_info['width']
    res_bbox = []
    res_segmentation = []
    res_labels = []
    for ann in annotations_list:
        box = ann['bbox']
        xmin = box[0]/W
        ymin = box[1]/H
        xmax = (box[0]+box[2])/W
        ymax = (box[1]+box[3])/H
        res_bbox.append([ymin,xmin,ymax,xmax])
        res_segmentation.append(ann['segmentation'])
        res_labels.append(ann['category_id'])

    if len(annotations_list)>0:
        return np.array(res_bbox),np.array(res_labels),np.array(res_segmentation)
    else:
        return np.zeros([0,4],dtype=np.float32),np.zeros([0],dtype=np.int32),np.zeros([0,H,W],dtype=np.uint8)


def get_files(data_dir, img_suffix=wmli.BASE_IMG_SUFFIX,keep_no_json_img=False):
    img_files = wmlu.recurse_get_filepath_in_dir(data_dir, suffix=img_suffix)
    res = []
    for img_file in img_files:
        json_file = wmlu.change_suffix(img_file, "json")
        if keep_no_json_img or os.path.exists(json_file):
            res.append((img_file, json_file))

    return res

def _get_shape_points(shape,circle_points_nr=20):
    shape_type = shape['shape_type']
    if shape_type == "polygon":
        return np.array(shape['points']).astype(np.int32)
    elif shape_type == "rectangle":
        points = np.array(shape['points']).astype(np.int32)
        x0 = np.min(points[:,0])
        x1 = np.max(points[:,0])
        y0 = np.min(points[:,1])
        y1 = np.max(points[:,1])
        n_points  = np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]]).astype(np.int32)
        return n_points
    elif shape_type == "circle":
        points = np.array(shape['points']).astype(np.int32)
        center = points[0]
        d = points[1]-points[0]
        r = math.sqrt(d[0]*d[0]+d[1]*d[1])
        points = points_on_circle(center=center,r=r,points_nr=circle_points_nr)
        return points
    else:
        print(f"WARNING: unsupport labelme shape type {shape_type}")
        return np.zeros([0,2],dtype=np.int32)
        


'''
output:
image_info: {'height','width'}
annotations_list: [{'bbox','segmentation','category_id','points_x','points_y'}' #bbox[xmin,ymin,width,height] absolute coordinate, 
'segmentation' [H,W], 全图
'''
def read_labelme_data(file_path,label_text_to_id=lambda x:int(x),mask_on=True,use_semantic=True,
                      use_polygon_mask=False,
                      circle_points_nr=20,
                      do_raise=False):
    if mask_on == False:
        use_semantic = False
    annotations_list = []
    image = {}
    try:
        with open(file_path,"r",encoding="utf-8") as f:
            data_str = f.read()
            json_data = json.loads(data_str)
            img_width = int(json_data["imageWidth"])
            img_height = int(json_data["imageHeight"])
            image["height"] = int(img_height)
            image["width"] = int(img_width)
            image["file_name"] = wmlu.base_name(file_path)
            for shape in json_data["shapes"]:
                all_points = _get_shape_points(shape,circle_points_nr=circle_points_nr).astype(np.int32) #[1,N,2]
                if len(all_points)<1:
                    continue
                points = np.transpose(all_points)
                x,y = np.vsplit(points,2)
                x = np.reshape(x,[-1])
                y = np.reshape(y,[-1])
                x = np.minimum(np.maximum(0,x),img_width-1)
                y = np.minimum(np.maximum(0,y),img_height-1)
                xmin = np.min(x)
                xmax = np.max(x)
                ymin = np.min(y)
                ymax = np.max(y)
                if mask_on:
                    all_points = np.expand_dims(all_points,axis=0)
                    if use_polygon_mask:
                        segmentation = WPolygonMaskItem(all_points,width=img_width,height=img_height)
                    else:
                        mask = np.zeros(shape=[img_height,img_width],dtype=np.uint8)
                        segmentation = cv.drawContours(mask,all_points,-1,color=(1),thickness=cv.FILLED)
                else:
                    segmentation = None

                flags = shape.get('flags',{})
                difficult = False
                for k,v in flags.items():
                    if not v:
                        continue
                    if k.lower() in ['crowd','ignore','difficult']:
                        difficult = True

                ori_label = shape['label']
                if "*" in ori_label:
                    difficult = True
                    ori_label = ori_label.replace("*","")

                if label_text_to_id is not None:
                    label = label_text_to_id(ori_label)
                else:
                    label = ori_label

                annotations_list.append({"bbox":(xmin,ymin,xmax-xmin+1,ymax-ymin+1),
                                         "segmentation":segmentation,
                                         "category_id":label,
                                         "points_x":x,
                                         "points_y":y,
                                         "difficult":difficult})
    except Exception as e:
        if do_raise:
            raise e
        image["height"] = 1
        image["width"] = 1
        image["file_name"] = wmlu.base_name(file_path)
        print(f"Read file {os.path.basename(file_path)} faild, info {e}.")
        annotations_list = []
        pass

    if use_semantic and not use_polygon_mask and mask_on:
        '''
        Each pixel only belong to one classes, and the latter annotation will overwrite the previous
        '''
        if len(annotations_list) > 1:
            mask = 1 - annotations_list[-1]['segmentation']
            for i in reversed(range(len(annotations_list) - 1)):
                annotations_list[i]['segmentation'] = np.logical_and(annotations_list[i]['segmentation'], mask)
                mask = np.logical_and(mask, 1 - annotations_list[i]['segmentation'])
    return image,annotations_list

def save_labelme_data(file_path,image_path,image,annotations_list,label_to_text=lambda x:str(x)):
    '''
    annotations_list[i]['segmentation'] [H,W] 全图mask
    '''
    data={}
    shapes = []
    data["version"] = "3.10.1"
    data["flags"] = {}
    for ann in annotations_list:
        shape = {}
        shape["label"] = label_to_text(ann["category_id"])
        #shape["line_color"]=None
        #shape["fill_color"]=None
        shape["shape_type"]="polygon"
        contours, hierarchy = cv.findContours(ann["segmentation"], cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        hierarchy = np.reshape(hierarchy,[-1,4]) 
        for he,cont in zip(hierarchy,contours):
            if he[-1]>=0 and cv.contourArea(cont) < cv.contourArea(contours[[he[-1]]]):
                continue
            points = cont
            if len(cont.shape)==3 and cont.shape[1]==1:
                points = np.squeeze(points,axis=1)
            points = points.tolist()
            shape["points"] = points
            shapes.append(copy.deepcopy(shape))

    data["shapes"] = shapes
    data["imagePath"] = os.path.basename(image_path)
    if image is not None:
        data["imageWidth"] = image["width"]
        data["imageHeight"] = image["height"]
    else:
        height,width = wmli.get_img_size(image_path)
        data["imageWidth"] = width
        data["imageHeight"] = height

    data["imageData"] = None
    with open(file_path,"w") as f:
        json.dump(data,f)

def save_labelme_datav2(file_path,image_path,image,annotations_list,label_to_text=lambda x:str(x)):
    '''
    mask 仅包含bboxes中的部分
    annotations_list[i]['bbox'] (x0,y0,x1,y1) 绝对坐标
    annotations_list[i]["segmentation"] (H,W), 仅包含bbox内部分
    '''
    data={}
    shapes = []
    data["version"] = "3.10.1"
    data["flags"] = {}
    if isinstance(label_to_text,dict):
        label_to_text = wmlu.MDict.from_dict(label_to_text)
    for ann in annotations_list:
        shape = {}
        if label_to_text is not None:
            shape["label"] = label_to_text(ann["category_id"])
        else:
            shape["label"] = ann["category_id"]
        #shape["line_color"]=None
        #shape["fill_color"]=None
        shape["shape_type"]="polygon"
        mask = ann["segmentation"]
        x0,y0,x1,y1 = ann['bbox']
        scale = np.reshape(np.array([(x1-x0)/mask.shape[1],(y1-y0)/mask.shape[0]],dtype=np.float32),[1,2])
        offset = np.reshape(np.array([x0,y0],dtype=np.float32),[1,2])

        contours, hierarchy = cv.findContours(ann["segmentation"], cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        hierarchy = np.reshape(hierarchy,[-1,4]) 
        for he,cont in zip(hierarchy,contours):
            if he[-1]>=0 and cv.contourArea(cont) < cv.contourArea(contours[[he[-1]]]):
                continue
            points = cont
            if len(cont.shape)==3 and cont.shape[1]==1:
                points = np.squeeze(points,axis=1)
            points = points*scale+offset
            points = points.astype(np.int32).tolist()
            if len(points)<=2:
                continue
            shape["points"] = points
            shapes.append(copy.deepcopy(shape))

    data["shapes"] = shapes
    data["imagePath"] = os.path.basename(image_path)
    data["imageWidth"] = image["width"]
    data["imageHeight"] = image["height"]
    data["imageData"] = None
    with open(file_path,"w") as f:
        json.dump(data,f)
'''
使用目标检测输出保存文件
'''
def save_labelme_datav3(file_path,image_path,image,labels,bboxes,masks,label_to_text=lambda x:str(x)):
    '''
    labels:[N]
    bboxes:[N,4](x0,y0,x1,y1), 绝对坐标
    masks:[N,h,w] 仅包含bbox内的部分
    '''
    annotatios_list = []
    for i in range(len(labels)):
        annotatios = {"category_id":labels[i],
        'segmentation':masks[i].astype(np.uint8),
        'bbox':bboxes[i]}
        annotatios_list.append(annotatios)
    save_labelme_datav2(file_path,image_path,image,annotatios_list,label_to_text=label_to_text)

def save_labelme_datav4(file_path,image_path,image,annotations_list,label_to_text=lambda x:str(x)):
    '''
    mask 仅包含bboxes中的部分
    annotations_list[i]['bbox'] (x0,y0,x1,y1) 绝对坐标
    annotations_list[i]["segmentation"] list[(N,2)] points
    '''
    data={}
    shapes = []
    data["version"] = "3.10.1"
    data["flags"] = {}
    if isinstance(label_to_text,dict):
        label_to_text = wmlu.MDict.from_dict(label_to_text)
    for ann in annotations_list:
        masks = ann["segmentation"]
        for mask in masks:
            shape = {}
            if label_to_text is not None:
                shape["label"] = label_to_text(ann["category_id"])
            else:
                shape["label"] = ann["category_id"]
            #shape["line_color"]=None
            #shape["fill_color"]=None
            shape["shape_type"]="polygon"
            if isinstance(mask,np.ndarray):
                mask = mask.tolist()
            shape["points"] = mask
            shapes.append(copy.deepcopy(shape))

    data["shapes"] = shapes
    data["imagePath"] = os.path.basename(image_path)
    data["imageWidth"] = image["width"]
    data["imageHeight"] = image["height"]
    data["imageData"] = None
    with open(file_path,"w") as f:
        json.dump(data,f)

def save_labelme_datav5(file_path,image_path,image,labels,bboxes,masks,label_to_text=lambda x:str(x)):
    '''
    labels:[N]
    bboxes:[N,4](x0,y0,x1,y1), 绝对坐标
    masks:WPolygonMasks
    '''
    annotatios_list = []
    for i in range(len(labels)):
        annotatios = {"category_id":labels[i],
        'segmentation':masks[i].points,
        'bbox':bboxes[i]}
        annotatios_list.append(annotatios)
    save_labelme_datav4(file_path,image_path,image,annotatios_list,label_to_text=label_to_text)

def save_labelme_datav6(file_path,image_path,masks,labels,image=None):
    '''
    masks: [N,H,W],整图mask
    '''
    data={}
    shapes = []
    data["version"] = "3.10.1"
    data["flags"] = {}
    if image is None:
        size = wmli.get_img_size(image_path)
        image = dict(width=size[1],height=size[0])
    for label,mask in zip(labels,masks):
        shape = {}
        shape["label"] = label
        #shape["line_color"]=None
        #shape["fill_color"]=None
        shape["shape_type"]="polygon"

        apoints,_ = findContours(mask.astype(np.uint8))
        for points in apoints:
            points = points.astype(np.int32).tolist()
            if len(points)<=2:
                continue
            shape["points"] = points
            shapes.append(copy.deepcopy(shape))

    data["shapes"] = shapes
    data["imagePath"] = os.path.basename(image_path)
    data["imageWidth"] = image["width"]
    data["imageHeight"] = image["height"]
    data["imageData"] = None
    with open(file_path,"w") as f:
        json.dump(data,f)

def save_labelme_points_data(file_path,image_path,image,points,labels):
    '''
    points: [N,2] (x,y)
    labels: [N]
    '''
    data={}
    shapes = []
    data["version"] = "4.2.9"
    data["flags"] = {}

    if image is None:
        h,w = wmli.get_img_size(image_path)
        image = dict(width=w,height=h)

    for point,label in zip(points,labels):
        shape = {}
        shape["label"] = str(label)
        shape['group_id'] = None
        shape["shape_type"]="point"
        shape["points"] = [[int(point[0]),int(point[1])]]
        flags = dict(ignore=False,difficult=False,crowd=False)
        shape['flags'] = flags
        shapes.append(shape)

    data["shapes"] = shapes
    data["imagePath"] = os.path.basename(image_path)
    data["imageWidth"] = image["width"]
    data["imageHeight"] = image["height"]
    data["imageData"] = None

    with open(file_path,"w") as f:
        json.dump(data,f)

def get_labels_and_bboxes(image,annotations_list,is_relative_coordinate=False):
    labels = []
    bboxes = []
    width = image["width"]
    height = image["height"]
    for ann in annotations_list:
        t_box = ann["bbox"]
        xmin = t_box[0]
        ymin = t_box[1]
        xmax = xmin+t_box[2]
        ymax = ymin+t_box[3]
        bboxes.append([ymin,xmin,ymax,xmax])
        labels.append(ann["category_id"])
    if len(bboxes)>0:
        bboxes = np.array(bboxes,dtype=np.float32)
    else:
        bboxes = np.zeros([0,4],dtype=np.float32)
    if is_relative_coordinate:
        div = np.array([[height,width,height,width]],dtype=np.float32)
        bboxes = bboxes/div
    return np.array(labels),bboxes

def get_labels_bboxes_and_masks(image,annotations_list):
    labels = []
    bboxes = []
    masks = []
    width = image["width"]
    height = image["height"]
    for ann in annotations_list:
        t_box = ann["bbox"]
        xmin = t_box[0]/width
        ymin = t_box[1]/height
        xmax = xmin+t_box[2]/width
        ymax = ymin+t_box[3]/height
        bboxes.append([ymin,xmin,ymax,xmax])
        labels.append(ann["category_id"])
        masks.append(ann["segmentation"])
    return np.array(labels),np.array(bboxes),np.array(masks)

'''
output:
[num_classes,h,w] or [num_classes-1,h,w], value is 0 or 1
'''
def get_masks(image,annotations_list,num_classes,no_background=True):
    width = image["width"]
    height = image["height"]
    if no_background:
        get_label = lambda x:max(0,x-1)
        res = np.zeros([num_classes-1,height,width],dtype=np.int32)
    else:
        get_label = lambda x:x
        res = np.zeros([num_classes,height,width],dtype=np.int32)

    for ann in annotations_list:
        mask = ann["segmentation"]
        label = get_label(ann["category_id"])
        res[label:label+1,:,:] = res[label:label+1,:,:]|np.expand_dims(mask,axis=0)

    return res

def get_image_size(image):
    width = image["width"]
    height = image["height"]
    return (height,width)

def save_labelme_datav1(file_path,image_path,image_data,annotations_list,label_to_text=lambda x:str(x)):
    wmli.imsave(image_path,image_data)
    image = {"width":image_data.shape[1],"height":image_data.shape[0]}
    save_labelme_data(file_path,image_path,image,annotations_list,label_to_text)

'''
获取标注box scale倍大小的扩展box(面积为scale*scale倍大)
box的中心点不变
'''
def get_expand_bboxes_in_annotations(annotations,scale=2):
   bboxes = [ann["bbox"] for ann in annotations]
   bboxes = odb.expand_bbox(bboxes,scale)
   return bboxes

'''
获取标注box 扩展为size大小的box
box的中心点不变
'''
def get_expand_bboxes_in_annotationsv2(annotations,size):
    bboxes = [ann["bbox"] for ann in annotations]
    bboxes = odb.expand_bbox_by_size(bboxes,size)
    return bboxes

def get_labels(annotations):
    labels = [ann["category_id"] for ann in annotations]
    return labels

'''
size:(h,w)
'''
def resize(image,annotations_list,img_data,size):
    res_image = copy.deepcopy(image)
    res_image["width"] = size[1]
    res_image["height"] = size[0]
    res_ann = []

    res_img_data = wmli.resize_img(img_data,size,keep_aspect_ratio=True)
    x_scale = res_img_data.shape[1]/img_data.shape[1]
    y_scale = res_img_data.shape[0]/img_data.shape[0]

    for ann in  annotations_list:
        bbox = (np.array(ann["bbox"])*np.array([x_scale,y_scale,x_scale,y_scale])).astype(np.int32)
        segmentation = wmli.resize_img(ann["segmentation"],size=size,keep_aspect_ratio=True,
                                      interpolation=cv2.INTER_NEAREST)
        category = copy.deepcopy(ann["category_id"])
        res_ann.append({"bbox":bbox,"segmentation":segmentation,"category_id":category})

    return res_image,res_ann,res_img_data

'''
从目标集中随机的选一个目标并从中截图
随机的概率可以通过weights指定
'''
def random_cut(image,annotations_list,img_data,size,weights=None,threshold=0.15):
    x_max = max(0,image["width"]-size[0])
    y_max = max(0,image["height"]-size[1])
    image_info = {}
    image_info["height"] =size[1]
    image_info["width"] =size[0]
    obj_ann_bboxes = get_expand_bboxes_in_annotations(annotations_list,2)
    labels = get_labels(annotations_list)
    if len(annotations_list)==0:
        return None,None,None
    count = 1
    while count<100:
        t_bbox = odb.random_bbox_in_bboxes(obj_ann_bboxes,size,weights,labels)
        t_bbox[1] = max(0,min(t_bbox[1],y_max))
        t_bbox[0] = max(0,min(t_bbox[0],x_max))
        rect = (t_bbox[1],t_bbox[0],t_bbox[1]+t_bbox[3],t_bbox[0]+t_bbox[2])
        new_image_info,new_annotations_list,new_image_data = cut(annotations_list,img_data,rect,threshold=threshold)
        if new_annotations_list is not None and len(new_annotations_list)>0:
            return (new_image_info,new_annotations_list,new_image_data)
        ++count

    return None,None,None

'''
size:[H,W]
在每一个标目标附近裁剪出一个子图
'''
def random_cutv1(image,annotations_list,img_data,size,threshold=0.15,resize_size=None):
    res = []
    x_max = max(0,image["width"]-size[0])
    y_max = max(0,image["height"]-size[1])
    image_info = {}
    image_info["height"] =size[1]
    image_info["width"] =size[0]
    obj_ann_bboxes = get_expand_bboxes_in_annotationsv2(annotations_list,[x//2 for x in size])
    if len(annotations_list)==0:
        return res

    for t_bbox in obj_ann_bboxes:
        t_bbox = list(t_bbox)
        t_bbox[1] = max(0,min(t_bbox[1],y_max))
        t_bbox[0] = max(0,min(t_bbox[0],x_max))
        t_bbox = odb.random_bbox_in_bbox(t_bbox,size)
        rect = (t_bbox[1],t_bbox[0],t_bbox[1]+t_bbox[3],t_bbox[0]+t_bbox[2])
        new_image_info,new_annotations_list,new_image_data = cut(annotations_list,img_data,rect,threshold=threshold)
        if new_annotations_list is not None and len(new_annotations_list)>0:
            if resize_size is not None:
                new_image_info, new_annotations_list, new_image_data = resize(new_image_info, new_annotations_list,
                                                                              new_image_data, resize_size)

            res.append((new_image_info,new_annotations_list,new_image_data))
    return res


'''
ref_bbox: [N,4] relative coordinate,[ymin,xmin,ymax,xmax]
'''
def random_cutv2(image,annotations_list,ref_bbox,img_data,size,weights=None):
    x_max = max(0,image["width"]-size[0])
    y_max = max(0,image["height"]-size[1])
    image_info = {}
    image_info["height"] =size[1]
    image_info["width"] =size[0]
    obj_ann_bboxes = []
    for bbox in ref_bbox:
        ymin = int(bbox[0]*image["height"])
        xmin = int(bbox[1]*image["width"])
        width = int((bbox[3]-bbox[1])*image["width"])
        height = int((bbox[2]-bbox[0])*image["height"])
        obj_ann_bboxes.append([xmin,ymin,width,height])
    labels = get_labels(annotations_list)
    t_bbox = odb.random_bbox_in_bboxes(obj_ann_bboxes,size,weights,labels)
    t_bbox[1] = max(0,min(t_bbox[1],y_max))
    t_bbox[0] = max(0,min(t_bbox[0],x_max))
    rect = (t_bbox[1],t_bbox[0],t_bbox[1]+t_bbox[3],t_bbox[0]+t_bbox[2])
    new_image_info,new_annotations_list,new_image_data = cut(annotations_list,img_data,rect,return_none_if_no_ann=False)
    return (new_image_info,new_annotations_list,new_image_data)


'''
image_data:[h,w,c]
bbox:[ymin,xmin,ymax,xmax)
output:
image_info: {'height','width'}
annotations_list: [{'bbox','segmentation','category_id'}' #bbox[xmin,ymin,width,height] absolute coordinate, 'segmentation' [H,W]
image_data:[H,W,3]
'''
def cut(annotations_list,img_data,bbox,threshold=0.15,return_none_if_no_ann=True):
    bbox = list(bbox)
    bbox[0] = max(0,bbox[0])
    bbox[1] = max(0,bbox[1])
    bbox[2] = min(bbox[2],img_data.shape[0])
    bbox[3] = min(bbox[3],img_data.shape[1])

    size = (bbox[3]-bbox[1],bbox[2]-bbox[0])
    new_annotations_list = []
    image_info = {}
    image_info["height"] =size[1]
    image_info["width"] =size[0]
    area = size[1]*size[0]
    image_info["file_name"] = f"IMG_L{bbox[1]:06}_T{bbox[0]:06}_W{bbox[3]-bbox[1]:06}_H{bbox[2]-bbox[0]:06}"
    for obj_ann in annotations_list:
        cnts,bboxes,ratios = odb.cut_contourv2(obj_ann["segmentation"],bbox)
        label = obj_ann["category_id"]
        if len(cnts)>0:
            for i,cnt in enumerate(cnts):
                ratio = ratios[i]
                t_bbox = odb.to_xyminwh(odb.bbox_of_contour(cnt))
                if ratio<threshold:
                    continue
                mask = np.zeros(shape=[size[1],size[0]],dtype=np.uint8)
                segmentation = cv.drawContours(mask,np.array([cnt]),-1,color=(1),thickness=cv.FILLED)
                new_annotations_list.append({"bbox":t_bbox,"segmentation":segmentation,"category_id":label})
    if (len(new_annotations_list)>0) or (not return_none_if_no_ann):
        return (image_info,new_annotations_list,wmli.sub_image(img_data,bbox))
    else:
        return None,None,None


            
def remove_instance(image,annotations_list,remove_pred_fn,default_value=[127, 127, 127]):
    res = []
    removed_image = np.ones_like(image) * np.array([[default_value]], dtype=np.uint8)

    chl = image.shape[-1]
    for ann in annotations_list:
        if remove_pred_fn(ann):
            mask = ann['segmentation']
            select = np.greater(mask, 0)
            select = np.expand_dims(select,axis=-1)
            select = np.tile(select, [1, 1, chl])
            image = np.where(select, removed_image, image)
        else:
            res.append(ann)
    
    return image,res

def read_labelme_kp_data(file_path,label_text_to_id=lambda x:int(x)):
    '''

    Args:
        file_path: json file path
        label_text_to_id: int f(string)

    Returns:
        labels:[N]
        points:[N,2]
    '''
    labels = []
    points = []
    image_info = {}
    with open(file_path,"r") as f:
        data = json.load(f)

    for d in data['shapes']:
        label = d['label']
        point = d['points'][0]
        if label_text_to_id is not None:
            label = label_text_to_id(label)
        labels.append(label)
        points.append(point)

    image_info['width'] = int(data['imageWidth'])
    image_info['height'] = int(data["imageHeight"])
    image_info['file_name'] = wmlu.base_name(data["imagePath"])

    return image_info,labels,points


def read_labelme_mckp_data(file_path,label_text_to_id=None,keep_no_json_img=False):
    '''

    Args:
        file_path: json file path
        label_text_to_id: int f(string)

    Returns:
        labels:[N]
        points:list of [N,2] points, [x,y]
    '''
    labels = []
    points = []
    image_info = {}

    kp_datas = wmlu.MDict(dtype=list)

    if os.path.exists(file_path):
        with open(file_path,"r") as f:
            data = json.load(f)
        for d in data['shapes']:
            label = d['label']
            point = np.reshape(np.array(d['points']),[-1,2])
            if label_text_to_id is not None:
                label = label_text_to_id(label)
            kp_datas[label.lower()].append(point)

    image_info[WIDTH] = int(data['imageWidth'])
    image_info[HEIGHT] = int(data["imageHeight"])
    image_info[FILENAME] = wmlu.base_name(data["imagePath"])
    image_info[FILEPATH] = data["imagePath"]

    for k,v in kp_datas.items():
        #v is [N,2]
        labels.append(k)
        points.append(np.concatenate(v,axis=0))

    return image_info,labels,points

