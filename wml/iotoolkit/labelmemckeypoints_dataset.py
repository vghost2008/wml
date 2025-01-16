#coding=utf-8
import numpy as np
import matplotlib.image as mpimg
import xml.etree.ElementTree as ET
from xml.dom.minidom import Document
import random
import os
import math
import wml.wml_utils
import logging
import shutil
from functools import partial
import wml.wml_utils as wmlu
import wml.img_utils as wmli
import copy
from easydict import EasyDict
from wml.object_detection2.standard_names import *
import pickle
from .common import *
import wml.object_detection2.bboxes as odb
from .pascal_voc_toolkit_fwd import *
from wml.wstructures import WMCKeypoints
from .labelme_toolkit_fwd import get_files,read_labelme_mckp_data


class LabelmeMCKeypointsDataset(object):
    def __init__(self, label_text2id=None, shuffle=False,
                 filter_empty_files=False,
                 resample_parameters=None,
                 keep_no_json_img=False,
                 ignore_case=True):
        '''

        :param label_text2id: trans a single label text to id:  int func(str)
        :param shuffle:
        :param image_sub_dir:
        :param xml_sub_dir:
        '''
        self.files = None
        self.shuffle = shuffle
        self.filter_empty_files = filter_empty_files
        self.keep_no_json_img = keep_no_json_img
        if isinstance(label_text2id,dict):
            if ignore_case:
                new_dict = dict([(k.lower(),v) for k,v in label_text2id.items()])
                self.label_text2id = partial(ignore_case_dict_label_text2id,dict_data=new_dict)
            else:
                self.label_text2id = partial(dict_label_text2id,dict_data=label_text2id)
        else:
            self.label_text2id = label_text2id

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


    def __len__(self):
        return len(self.files)

    def __getitem__(self,idx):
        try:
            fs = self.files[idx]
        except Exception as e:
            print(f"ERROR: {e} {self.files[idx]}")
            print(self.files)
        #print(xml_file)
        try:
            data = self.read_one_file(fs)
            labels_names = data[GT_LABELS]
            img_info = data[IMG_INFO]
            if self.label_text2id is not None:
                labels = [self.label_text2id(x) for x in labels_names]
                keypoints = data[GT_KEYPOINTS]
                keep = [x is not None for x in labels]
                labels = list(filter(lambda x:x is not None,labels))
                if len(labels)==0 or not isinstance(labels[0],(str,bytes)):
                    labels = np.array(labels,dtype=np.int32)
                keypoints = [keypoints[i] for i in np.where(keep)[0]]
                data[GT_LABELS] = labels
                data[GT_KEYPOINTS] = WMCKeypoints(keypoints,width=img_info[WIDTH],height=img_info[HEIGHT])
            else:
                keypoints = data[GT_KEYPOINTS]
                data[GT_KEYPOINTS] = WMCKeypoints(keypoints,width=img_info[WIDTH],height=img_info[HEIGHT])

        except Exception as e:
            print(f"Read {fs} {e} faild.")
            return None
        return EasyDict(data)

    def read_data(self,dir_path,img_suffix=wmli.BASE_IMG_SUFFIX):
        if isinstance(dir_path,str):
            print(f"Read {dir_path}")
            if not os.path.exists(dir_path):
                print(f"Data path {dir_path} not exists.")
                return False
            self.files = get_files(dir_path,img_suffix=img_suffix,keep_no_json_img=self.keep_no_json_img)
        elif isinstance(dir_path,(list,tuple)) and isinstance(dir_path[0],(str,bytes)) and os.path.isdir(dir_path[0]):
            self.files = []
            for dp in dir_path:
                self.files.extend(get_files(dp,
                                            img_suffix=img_suffix,
                                            keep_no_json_img=self.keep_no_json_img))
        else:
            self.files = dir_path
        if self.filter_empty_files and self.label_text2id:
            self.files = self.apply_filter_empty_files(self.files)
        if self.resample_parameters is not None and self.label_text2id:
            self.files = self.resample(self.files)
        
        if len(self.files) == 0:
            return False

        if self.shuffle:
            random.shuffle(self.files)

        return True

    def read_one_file(self,data):
        img_file,json_file = data 
        image_info,labels,points = read_labelme_mckp_data(json_file,keep_no_json_img=self.keep_no_json_img)
        image_info[FILEPATH] = img_file
        datas = {}
        datas[IMG_INFO] = image_info
        datas[GT_LABELS] = labels
        datas[GT_KEYPOINTS] = points
        with open(img_file,"rb") as f:
            img_data = f.read()
        datas[IMAGE] = img_data
        return datas

    def apply_filter_empty_files(self,files):
        new_files = []
        for fs in files:
            data = self.read_one_file(fs)
            labels_name = data[GT_LABELS]
            labels = [self.label_text2id(x) for x in labels_name]
            is_none = [x is None for x in labels]
            if not all(is_none):
                new_files.append(fs)
            else:
                print(f"File {fs} is empty, labels names {labels_name}, labels {labels}")

        return new_files
    
    def resample(self,files):
        all_labels = []
        for fs in files:
            data = self.read_one_file(fs)
            labels_name = data[GT_LABELS]
            labels = [self.label_text2id(x) for x in labels_name]
            all_labels.append(labels)

        return resample(files,all_labels,self.resample_parameters)


    def get_items(self):
        '''
        :return: 
        full_path,img_size,category_ids,category_names,boxes,binary_masks,area,is_crowd,num_annotations_skipped
        '''
        for i in range(len(self.files)):
            yield self.__getitem__(i)