import wml_utils as wmlu
import os
import json
import numpy as np
import cv2 as cv
import copy
import img_utils as wmli
import random
import matplotlib.pyplot as plt
import sys
import cv2
from object_detection2.standard_names import *
import object_detection2.bboxes as odb
from functools import partial
from .common import *
from .labelme_toolkit_fwd import *
import glob

class LabelMeBase(object):
    def __init__(self,label_text2id=None,shuffle=False,absolute_coord=True,
                 filter_empty_files=False,
                 filter_error=False,
                 resample_parameters=None,
                 use_polygon_mask=False,
                 read_data_kwargs={'circle_points_nr':20}):
        '''
        label_text2id: func(name)->int
        '''
        self.files = None
        if isinstance(label_text2id,dict):
            self.label_text2id = partial(ignore_case_dict_label_text2id,
                    dict_data=wmlu.trans_dict_key2lower(label_text2id))
        else:
            self.label_text2id = label_text2id
        self.shuffle = shuffle
        self.absolute_coord = absolute_coord
        self.filter_empty_files = filter_empty_files
        self.filter_error = filter_error
        self.read_data_kwargs = read_data_kwargs
        self.use_polygon_mask = use_polygon_mask
        if resample_parameters is not None:
            self.resample_parameters = {}
            for k,v in resample_parameters.items():
                if isinstance(k,(str,bytes)):
                    k = self.label_text2id(k)
                self.resample_parameters[k] = v
            print("resample parameters")
            wmlu.show_dict(self.resample_parameters)
        else:
            self.resample_parameters = None
        
    def read_data(self,dir_path,img_suffix=".jpg;;.bmp;;.png;;.jpeg"):
        _files = get_files(dir_path,img_suffix=img_suffix)
        if self.filter_empty_files and self.label_text2id:
            _files = self.apply_filter_empty_files(_files)
        elif self.filter_error:
            _files = self.apply_filter_error_files(_files)
        if self.resample_parameters is not None and self.label_text2id:
            _files = self.resample(_files)
        
        self.files = _files

        print(f"Total find {len(self.files)} in {dir_path}")
        if self.shuffle:
            random.shuffle(self.files)
        print("Files")
        wmlu.show_list(self.files[:100])
        if len(self.files)>100:
            print("...")

    def apply_filter_empty_files(self,files):
        new_files = []
        for fs in files:
            try:
                img_file,json_file = fs
                image, annotations_list = read_labelme_data(json_file, None,use_semantic=True,mask_on=False,
                                                            use_polygon_mask=self.use_polygon_mask,
                                                            **self.read_data_kwargs)
                labels_names,bboxes = get_labels_and_bboxes(image,annotations_list,is_relative_coordinate=not self.absolute_coord)
                labels = [self.label_text2id(x) for x in labels_names]
                is_none = [x is None for x in labels]
                if not all(is_none):
                    new_files.append(fs)
                else:
                    print(f"File {json_file} is empty, remove from dataset, labels names {labels_names}, labels {labels}")
            except Exception as e:
                print(f"Read {json_file} faild, info: {e}.")
                pass

        return new_files

    def apply_filter_error_files(self,files):
        new_files = []
        for fs in files:
            try:
                img_file,json_file = fs
                image, annotations_list = read_labelme_data(json_file, None,use_semantic=True,mask_on=False,
                                                            use_polygon_mask=self.use_polygon_mask,
                                                            do_raise=True,
                                                            **self.read_data_kwargs)
                labels_names,bboxes = get_labels_and_bboxes(image,annotations_list,is_relative_coordinate=not self.absolute_coord)
                if self.label_text2id:
                    labels = [self.label_text2id(x) for x in labels_names] #测试转label是否有误
                new_files.append(fs)
            except Exception as e:
                print(f"Read {json_file} faild, info: {e}.")
                pass

        return new_files

    def resample(self,files):
        all_labels = []
        for fs in files:
            try:
                img_file,json_file = fs
                image, annotations_list = read_labelme_data(json_file, None,use_semantic=True,mask_on=False,
                                                            use_polygon_mask=self.use_polygon_mask,
                                                            **self.read_data_kwargs)
                labels_names,bboxes = get_labels_and_bboxes(image,annotations_list,is_relative_coordinate=not self.absolute_coord)
                labels = [self.label_text2id(x) for x in labels_names]
                all_labels.append(labels)
            except Exception as e:
                print(f"Labelme resample error: {e}")

        return resample(files,all_labels,self.resample_parameters)

    def __len__(self):
        return len(self.files)

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
            image, annotations_list = read_labelme_data(json_file, None,**self.read_data_kwargs)
            labels_names,bboxes = get_labels_and_bboxes(image,annotations_list,is_relative_coordinate=not self.absolute_coord)
            labels = [self.label_text2id(x) for x in labels_names]
            yield DetBboxesData(img_file,[image['height'],image['width']],labels, bboxes,  None)