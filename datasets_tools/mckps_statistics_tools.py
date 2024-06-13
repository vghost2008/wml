#coding=utf-8
import sys
import os
import object_detection2.npod_toolkit as npod
import wml_utils
import matplotlib.pyplot as plt
import numpy as np
import math
import object_detection2.visualization as odv
import img_utils as wmli
from iotoolkit.pascal_voc_toolkit import PascalVOCData,read_voc_xml
from iotoolkit.coco_toolkit import COCOData
from iotoolkit.labelme_toolkit import LabelMeData
from iotoolkit.fast_labelme import FastLabelMeData
import object_detection2.bboxes as odb 
import pandas as pd
import wml_utils as wmlu
from iotoolkit.mckeypoints_statistics import statistics_mckeypoints
from sklearn.cluster import KMeans
from functools import partial
from argparse import ArgumentParser
from itertools import count
from iotoolkit.object365v2_toolkit import Object365V2
from object_detection2.data_process_toolkit import remove_class
from collections import OrderedDict
from iotoolkit.bboxes_statistics import *
from iotoolkit.labelmemckeypoints_dataset import LabelmeMCKeypointsDataset


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('src_dir', type=str, help='source video directory')
    parser.add_argument('--type', type=str, default="LabelmeMCKeypointsDataset",help='dataset type')
    parser.add_argument('--labels', nargs="+",type=str,default=[],help='Config file')
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    nr = 100
    '''def trans_img_long_size(img_size):
        if img_size[0]<img_size[1]:
            k = 512/img_size[1]
        else:
            k = 512 / img_size[0]
        return [k*img_size[0],k*img_size[1]]
    
    def trans_img_short_size(img_size,min_size=640):
        if img_size[0]<img_size[1]:
            k = min_size/img_size[0]
        else:
            k = min_size/ img_size[1]
        return [k*img_size[0],k*img_size[1]]'''
    args = parse_args()
    data_dir = args.src_dir
    dataset_type = args.type
    if dataset_type == "LabelmeMCKeypointsDataset":
        dataset = LabelmeMCKeypointsDataset(label_text2id=None,shuffle=False)
        dataset.read_data(args.src_dir,img_suffix=wmli.BASE_IMG_SUFFIX)
    else:
        print(f"Unknow dataset type {dataset_type}")
    
    statistics_mckeypoints(dataset)
