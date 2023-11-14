#coding=utf-8
import numpy as np
import matplotlib.image as mpimg
import xml.etree.ElementTree as ET
from xml.dom.minidom import Document
import random
import os
import math
import wml_utils
import logging
import shutil
from functools import partial
import wml_utils as wmlu
import img_utils as wmli
import copy
from .common import *
import object_detection2.bboxes as odb
from .pascal_voc_toolkit_fwd import *


class PascalVOCData(object):
    def __init__(self, label_text2id=None, shuffle=False,image_sub_dir=None,xml_sub_dir=None,
                 has_probs=False,
                 absolute_coord=False,
                 filter_empty_files=False,
                 resample_parameters=None,
                 ignore_case=True):
        '''

        :param label_text2id: trans a single label text to id:  int func(str)
        :param shuffle:
        :param image_sub_dir:
        :param xml_sub_dir:
        '''
        self.files = None
        self.shuffle = shuffle
        self.xml_sub_dir = xml_sub_dir
        self.image_sub_dir = image_sub_dir
        self.has_probs = has_probs
        self.absolute_coord = absolute_coord
        self.filter_empty_files = filter_empty_files
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
            img_file,xml_file = self.files[idx]
        except Exception as e:
            print(f"ERROR: {e} {self.files[idx]}")
            print(self.files)
        #print(xml_file)
        if not os.path.exists(xml_file):
            return img_file,None,np.zeros([0],dtype=np.int32),[],np.zeros([0,4],dtype=np.float32),None,None,None,None
        try:
            data = read_voc_xml(xml_file,
                                adjust=None,
                                aspect_range=None,
                                has_probs=self.has_probs,
                                absolute_coord=self.absolute_coord)
            shape, bboxes, labels_names, difficult, truncated,probs = data

            if self.label_text2id is not None:
                labels = [self.label_text2id(x) for x in labels_names]
                keep = [x is not None for x in labels]
                labels = [x if x is not None else -1 for x in labels]
                labels = np.array(labels,dtype=np.int32)
                labels = labels[keep]
                bboxes = bboxes[keep]
                labels_names = np.array(labels_names)[keep]
            else:
                labels = None

        except Exception as e:
            print(f"Read {xml_file} {e} faild.")
            return img_file,None,np.zeros([0],dtype=np.int32),[],np.zeros([0,4],dtype=np.float32),None,None,None,None
        #使用difficult表示is_crowd
        return img_file, shape[:2],labels, labels_names, bboxes, None, None, difficult, probs

    def read_data(self,dir_path,silent=False,img_suffix=".jpg",check_xml_file=True):
        if isinstance(dir_path,str):
            print(f"Read {dir_path}")
            if not os.path.exists(dir_path):
                print(f"Data path {dir_path} not exists.")
                return False
            self.files = getVOCFiles(dir_path,image_sub_dir=self.image_sub_dir,
                                 xml_sub_dir=self.xml_sub_dir,
                                 img_suffix=img_suffix,
                                 silent=silent,
                                 check_xml_file=check_xml_file)
        elif isinstance(dir_path,(list,tuple)) and isinstance(dir_path[0],(str,bytes)) and os.path.isdir(dir_path[0]):
            self.files = self.get_files_from_dirs(dir_path,
                              silent=silent,img_suffix=img_suffix,check_xml_file=check_xml_file)
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

    def get_files_from_dirs(self,dirs,silent=False,img_suffix=".jpg",check_xml_file=True):
        all_files = []
        for dir_path in dirs:
            files = getVOCFiles(dir_path,image_sub_dir=self.image_sub_dir,
                                 xml_sub_dir=self.xml_sub_dir,
                                 img_suffix=img_suffix,
                                 silent=silent,
                                 check_xml_file=check_xml_file)
            all_files.extend(files)

        return all_files
    
    def apply_filter_empty_files(self,files):
        new_files = []
        for fs in files:
            img_file,xml_file = fs
            data = read_voc_xml(xml_file,
                                adjust=None,
                                aspect_range=None,
                                has_probs=self.has_probs,
                                absolute_coord=self.absolute_coord)
            shape, bboxes, labels_names, difficult, truncated,probs = data
            labels = [self.label_text2id(x) for x in labels_names]
            is_none = [x is None for x in labels]
            if not all(is_none):
                new_files.append(fs)
            else:
                print(f"File {xml_file} is empty, labels names {labels_names}, labels {labels}")

        return new_files
    
    def resample(self,files):
        all_labels = []
        for fs in files:
            img_file,xml_file = fs
            data = read_voc_xml(xml_file,
                                adjust=None,
                                aspect_range=None,
                                has_probs=self.has_probs,
                                absolute_coord=self.absolute_coord)
            shape, bboxes, labels_names, difficult, truncated,probs = data
            labels = [self.label_text2id(x) for x in labels_names]
            all_labels.append(labels)

        return resample(files,all_labels,self.resample_parameters)


    def get_items(self):
        '''
        :return: 
        full_path,img_size,category_ids,category_names,boxes,binary_masks,area,is_crowd,num_annotations_skipped
        '''
        for i in range(len(self.files)):
            yield self.__getitem__(i)

    def get_boxes_items(self):
        '''
        :return: 
        full_path,img_size,category_ids,boxes,is_crowd
        '''
        for img_file, xml_file in self.files:
            shape, bboxes, labels_names, difficult, truncated,probs = read_voc_xml(xml_file,
                                                                             adjust=None,
                                                                             aspect_range=None,
                                                                             has_probs=False)
            labels = [self.label_text2id(x) for x in labels_names]
            #使用difficult表示is_crowd
            yield img_file,shape[:2], labels, bboxes, difficult

if __name__ == "__main__":
    #data_statistics("/home/vghost/ai/mldata/qualitycontrol/rdatasv3")
    import object_detection2.visualization as odv
    import img_utils as wmli
    import matplotlib.pyplot as plt

    text = []
    for i in range(ord('a'), ord('z') + 1):
        text.append(chr(i))
    for i in range(ord('A'), ord('Z') + 1):
        text.append(chr(i))
    for i in range(ord('0'), ord('9') + 1):
        text.append(chr(i))
    '''
    0:53
    1:54
    ...
    9:62
    '''
    text.append('/')
    text.append('\\')
    text.append('-')
    text.append('+')
    text.append(":")
    text.append("WORD")
    text.append("WD0")  # up
    text.append("WD3")  # right
    text.append("WD1")  # down
    text.append("WD2")  # left
    text.append("##")  # left

    text_to_id = {}
    id_to_text = {}

    for i, t in enumerate(text):
        text_to_id[t] = i + 1
        id_to_text[i + 1] = t

    text_to_id[" "] = 69

    def name_to_id(name):
        return text_to_id[name]

    data = PascalVOCData(label_text2id=name_to_id,shuffle=True)
    data.read_data("/home/vghost/ai/mldata2/ocrdata/rdatasv20/train")
    MIN_IMG_SIZE = 768
    for x in data.get_items():
        full_path, category_ids, category_names, boxes, binary_mask, area, is_crowd, num_annotations_skipped = x
        img = wmli.imread(full_path)
        if img.shape[0]<MIN_IMG_SIZE or img.shape[1]<MIN_IMG_SIZE:
            img = wmli.resize_img(img,[MIN_IMG_SIZE,MIN_IMG_SIZE],keep_aspect_ratio=True)


        def text_fn(classes, scores):
            return id_to_text[classes]

        odv.bboxes_draw_on_imgv2(
            img=img, classes=category_ids, scores=None, bboxes=boxes, color_fn=None,
            text_fn=text_fn, thickness=2,
            show_text=True,
            fontScale=0.8)
        plt.figure()
        plt.imshow(img)
        plt.show()
