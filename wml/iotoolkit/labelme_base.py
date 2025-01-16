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
from .base_dataset import BaseDataset

class LabelMeBase(BaseDataset):
    def __init__(self,label_text2id=None,shuffle=True,absolute_coord=True,
                 filter_empty_files=False,
                 filter_error=False,
                 resample_parameters=None,
                 use_polygon_mask=False,
                 silent=False,
                 keep_no_ann_imgs=False,
                 mask_on=True,
                 read_data_kwargs={'circle_points_nr':20}):
        '''
        label_text2id: func(name)->int
        '''
        self.files = None
        super().__init__(label_text2id=label_text2id,
                          filter_empty_files=filter_empty_files,
                          filter_error=filter_error,
                          resample_parameters=resample_parameters,
                          shuffle=shuffle,
                          silent=silent,
                          absolute_coord=absolute_coord,
                          keep_no_ann_imgs=keep_no_ann_imgs,
                          mask_on=mask_on)
        self.read_data_kwargs = read_data_kwargs
        self.use_polygon_mask = use_polygon_mask

    def find_files_in_dir(self,dir_path,img_suffix=".jpg;;.bmp;;.png;;.jpeg"):
        files = get_files(dir_path,img_suffix=img_suffix,keep_no_json_img=self.keep_no_ann_imgs)
        return files

    def get_labels(self,fs):
        img_file,json_file = fs
        image, annotations_list = read_labelme_data(json_file, None,use_semantic=True,mask_on=False,
                                                    use_polygon_mask=True,
                                                    do_raise=True,
                                                    **self.read_data_kwargs)
        labels_names,bboxes = get_labels_and_bboxes(image,annotations_list,is_relative_coordinate=not self.absolute_coord)
        if self.label_text2id:
            labels = [self.label_text2id(x) for x in labels_names] #测试转label是否有误
        else:
            labels = None
        
        return labels,labels_names
    
    def get_ann_info(self,idx):
        img_file,json_file = self.files[idx]
        with open(json_file,"r",encoding="gb18030") as f:
            data_str = f.read()
            image = {}
            try:
                json_data = json.loads(data_str)
                img_width = int(json_data["imageWidth"])
                img_height = int(json_data["imageHeight"])
                image["height"] = int(img_height)
                image["width"] = int(img_width)
                image["file_name"] = wmlu.base_name(json_file)
            except:
                pass
        
        return image
    
    def get_items(self):
        '''
        :return: 
        full_path,img_size,category_ids,category_names,boxes,binary_masks,area,is_crowd,num_annotations_skipped
        '''
        for i in range(len(self.files)):
            sys.stdout.write('\r>> read data %d/%d' % (i + 1, len(self.files)))
            sys.stdout.flush()
            yield self.__getitem__(i)

    def get_boxes_items(self):
        '''
        :return: 
        full_path,img_size,category_ids,boxes,is_crowd
        '''
        for i,(img_file, json_file) in enumerate(self.files):
            sys.stdout.write('\r>> read data %d/%d' % (i + 1, len(self.files)))
            sys.stdout.flush()
            image, annotations_list = read_labelme_data(json_file, None,mask_on=False,**self.read_data_kwargs)
            labels_names,bboxes = get_labels_and_bboxes(image,annotations_list,is_relative_coordinate=not self.absolute_coord)
            labels = [self.label_text2id(x) for x in labels_names]
            yield DetBboxesData(img_file,[image['height'],image['width']],labels, bboxes,  None)