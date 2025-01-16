#coding=utf-8
import numpy as np
import matplotlib.image as mpimg
import xml.etree.ElementTree as ET
from xml.dom.minidom import Document
import random
import os
import math
import logging
import shutil
from functools import partial
import wml.wml_utils as wmlu
import wml.img_utils as wmli
import copy
from .common import *
import wml.object_detection2.bboxes as odb
from .pascal_voc_toolkit_fwd import *
from .base_dataset import BaseDataset


class PascalVOCData(BaseDataset):
    def __init__(self, label_text2id=None, shuffle=False,image_sub_dir=None,xml_sub_dir=None,
                 has_probs=False,
                 absolute_coord=False,
                 filter_error=False,
                 silent=False,
                 filter_empty_files=False,
                 keep_no_ann_imgs=False,
                 resample_parameters=None,
                 ignore_case=True):
        '''

        :param label_text2id: trans a single label text to id:  int func(str)
        :param shuffle:
        :param image_sub_dir:
        :param xml_sub_dir:
        '''
        self.files = None
        super().__init__(label_text2id=label_text2id,
                          filter_empty_files=filter_empty_files,
                          filter_error=filter_error,
                          resample_parameters=resample_parameters,
                          shuffle=shuffle,
                          silent=silent,
                          absolute_coord=absolute_coord,
                          keep_no_ann_imgs=keep_no_ann_imgs)
        self.xml_sub_dir = xml_sub_dir
        self.image_sub_dir = image_sub_dir
        self.has_probs = has_probs

    def __getitem__(self,idx):
        try:
            img_file,xml_file = self.files[idx]
        except Exception as e:
            print(f"ERROR: {e} {self.files[idx]}")
            print(self.files)
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
                difficult = np.array(difficult,dtype=np.int32)
                labels = labels[keep]
                bboxes = bboxes[keep]
                difficult = difficult[keep]
                labels_names = np.array(labels_names)[keep]
            else:
                labels = None

        except Exception as e:
            print(f"Read {xml_file} {e} faild.")
            return DetData(img_file,None,np.zeros([0],dtype=np.int32),[],np.zeros([0,4],dtype=np.float32),None,None,None,None)
        #使用difficult表示is_crowd
        return DetData(img_file, shape[:2],labels, labels_names, bboxes, None, None, difficult, probs)

    def find_files_in_dir(self,dir_path,img_suffix=".jpg"):
        if not os.path.exists(dir_path):
            print(f"Data path {dir_path} not exists.")
            return False
        files = getVOCFiles(dir_path,image_sub_dir=self.image_sub_dir,
                             xml_sub_dir=self.xml_sub_dir,
                             img_suffix=img_suffix,
                             silent=self.silent,
                             check_xml_file=not self.keep_no_ann_imgs)
        
        return files

    def get_labels(self,fs):
        img_file,xml_file = fs
        data = read_voc_xml(xml_file,
                            adjust=None,
                            aspect_range=None,
                            has_probs=self.has_probs,
                            absolute_coord=self.absolute_coord)
        shape, bboxes, labels_names, difficult, truncated,probs = data
        labels = [self.label_text2id(x) for x in labels_names]
        return labels,labels_names
    
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
            yield DetBboxesData(img_file,shape[:2], labels, bboxes, difficult)

if __name__ == "__main__":
    #data_statistics("/home/vghost/ai/mldata/qualitycontrol/rdatasv3")
    import wml.object_detection2.visualization as odv
    import wml.img_utils as wmli
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
