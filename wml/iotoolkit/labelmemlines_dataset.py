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
from wml.wstructures import WMLinesItem,WMLines
from .labelme_toolkit_fwd import get_files,read_labelme_mlines_data
from .base_dataset import BaseDataset
import traceback


class LabelmeMLinesDataset(BaseDataset):
    def __init__(self, label_text2id=None, shuffle=False,
                 filter_empty_files=False,
                 filter_error=False,
                 resample_parameters=None,
                 silent=True,
                 keep_no_ann_imgs=False,
                 ignore_case=True):
        '''

        :param label_text2id: trans a single label text to id:  int func(str)
        :param shuffle:
        :param image_sub_dir:
        :param xml_sub_dir:
        '''
        super().__init__(label_text2id=label_text2id,
                         filter_empty_files=filter_empty_files,
                         filter_error=filter_error,
                         resample_parameters=resample_parameters,
                         shuffle=shuffle,
                         silent=silent,
                         keep_no_ann_imgs=keep_no_ann_imgs,
                         )
        self.files = None


    def find_files_in_dir(self,dir_path,img_suffix=wmli.BASE_IMG_SUFFIX):
        return get_files(dir_path,img_suffix)
    

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
                keypoints = data[GT_LINES]
                keep = [x is not None for x in labels]
                labels = list(filter(lambda x:x is not None,labels))
                if len(labels)==0 or not isinstance(labels[0],(str,bytes)):
                    labels = np.array(labels,dtype=np.int32)
                keypoints = [keypoints[i] for i in np.where(keep)[0]]
                data[GT_LABELS] = labels
            else:
                keypoints = data[GT_LINES]

            data[GT_LINES] = WMLines(keypoints,width=img_info[WIDTH],height=img_info[HEIGHT])

        except Exception as e:
            print(f"Read {fs} {e} faild.")
            traceback.print_exc()
            return None
        return EasyDict(data)


    def read_one_file(self,data):
        img_file,json_file = data 
        image_info,labels,lines = read_labelme_mlines_data(json_file,keep_no_json_img=self.keep_no_ann_imgs)
        if image_info[WIDTH]<3 or image_info[HEIGHT]<3:
            img_size = wmli.get_img_size(img_file)
            image_info[HEIGHT] = img_size[0]
            image_info[WIDTH] = img_size[1]
        image_info[FILEPATH] = img_file
        datas = {}
        datas[IMG_INFO] = image_info
        datas[GT_LABELS] = labels
        datas[GT_LINES] = lines
        #with open(img_file,"rb") as f:
            #img_data = f.read()
        datas[IMAGE] = img_file
        return datas


    def get_items(self):
        '''
        :return: 
        full_path,img_size,category_ids,category_names,boxes,binary_masks,area,is_crowd,num_annotations_skipped
        '''
        for i in range(len(self.files)):
            yield self.__getitem__(i)

    def get_labels(self,fs):
        #return labels,labels_names
        image_info,labels_names,lines = read_labelme_mlines_data(fs,keep_no_json_img=self.keep_no_ann_imgs)
        if self.label_text2id is not None:
            labels= [self.label_text2id(x) for x in labels_names]
        else:
            labels = None
        
        return labels,labels_names