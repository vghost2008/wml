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
from .common import *
from .labelme_toolkit_fwd import *
import glob
from .labelme_base import LabelMeBase

class LabelMeData(LabelMeBase):
    def __init__(self,label_text2id=None,shuffle=True,absolute_coord=True,
                 filter_empty_files=False,
                 resample_parameters=None,
                 use_polygon_mask=False,
                 read_data_kwargs={'circle_points_nr':20},**kwargs):
        '''
        label_text2id: func(name)->int
        '''
        super().__init__(label_text2id=label_text2id,
                         shuffle=shuffle,
                         absolute_coord=absolute_coord,
                         filter_empty_files=filter_empty_files,
                         resample_parameters=resample_parameters,
                         read_data_kwargs=read_data_kwargs,**kwargs)
        self.use_polygon_mask = use_polygon_mask

    def __getitem__(self,idx):
        '''
        :return: 
        full_path,img_size,category_ids,category_names,boxes,binary_masks,area,is_crowd,num_annotations_skipped
        binary_masks:[N,H,W],全图
        bboxes:[N,4] (ymin,xmin,ymax,xmax)
        '''
        img_file, json_file = self.files[idx]
        image, annotations_list = read_labelme_data(json_file, None,mask_on=self.mask_on,use_semantic=True,
                                                    use_polygon_mask=self.use_polygon_mask,
                                                    **self.read_data_kwargs)
        labels_names,bboxes = get_labels_and_bboxes(image,annotations_list,is_relative_coordinate=not self.absolute_coord)
        difficult = np.array([v['difficult'] for v in annotations_list],dtype=bool)
        masks = [ann["segmentation"] for ann in annotations_list] if self.mask_on else None
        img_height = image['height']
        img_width = image['width']

        if not self.mask_on:
            pass
        elif self.use_polygon_mask:
            masks = WPolygonMasks(masks,width=img_width,height=img_height)
        elif len(masks)>0:
            try:
                masks = np.stack(masks,axis=0)
            except:
                masks = np.zeros(shape=[0,img_height,img_width],dtype=np.uint8)
        else:
            masks = np.zeros(shape=[len(labels_names),img_height,img_width],dtype=np.uint8)

        
        if self.label_text2id is not None:
            labels = [self.label_text2id(x) for x in labels_names]
            keep = [x is not None for x in labels]
            labels = [x if x is not None else -1 for x in labels]
            labels = np.array(labels,dtype=np.int32)
            labels = labels[keep]
            bboxes = bboxes[keep]
            if masks is not None:
                masks = masks[keep]
            difficult = difficult[keep]
            labels_names = np.array(labels_names)[keep]
        else:
            labels = None
            
        return DetData(img_file, [image['height'],image['width']],labels, labels_names, bboxes, masks, None, difficult,None)
    
            
if __name__ == "__main__":
    #data_statistics("/home/vghost/ai/mldata/qualitycontrol/rdatasv3")
    import wml.img_utils as wmli
    import wml.object_detection2.visualization as odv
    import matplotlib.pyplot as plt
    ID_TO_TEXT = {1:{"id":1,"name":"a"},2:{"id":2,"name":"b"},3:{"id":3,"name":"c"}}
    NAME_TO_ID = {}
    for k,v in ID_TO_TEXT.items():
        NAME_TO_ID[v["name"]] = v["id"]
    def name_to_id(name):
        return NAME_TO_ID[name]

    data = LabelMeData(label_text2id=name_to_id,shuffle=True)
    #data.read_data("/data/mldata/qualitycontrol/rdatasv5_splited/rdatasv5")
    #data.read_data("/home/vghost/ai/mldata2/qualitycontrol/rdatav10_preproc")
    #data.read_data("/home/vghost/ai/mldata2/qualitycontrol/rdatasv10_neg_preproc")
    data.read_data("/home/vghost/ai/mldata2/qualitycontrol/rdatasv10_1x_neg_preproc")
    #data.read_data("/home/vghost/ai/mldata2/qualitycontrol/x")
    for x in data.get_items():
        full_path, img_info,category_ids, category_names, boxes, binary_mask, area, is_crowd, num_annotations_skipped = x
        img = wmli.imread(full_path)


        def text_fn(classes, scores):
            return ID_TO_TEXT[classes]['name']

        odv.draw_bboxes_and_maskv2(
            img=img, classes=category_ids, scores=None, bboxes=boxes, masks=binary_mask, color_fn=None,
            text_fn=text_fn, thickness=4,
            show_text=True,
            fontScale=0.8)
        plt.figure()
        plt.imshow(img)
        plt.show()
