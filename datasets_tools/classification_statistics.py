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
from iotoolkit.mapillary_vistas_toolkit import MapillaryVistasData
from iotoolkit import get_auto_dataset_suffix
from sklearn.cluster import KMeans
from functools import partial
from argparse import ArgumentParser
from itertools import count
from iotoolkit.object365v2_toolkit import Object365V2
from object_detection2.data_process_toolkit import remove_class
from collections import OrderedDict
from iotoolkit.image_folder import *
from iotoolkit.classification_data_statistics import labels_statistics
from thirdparty.registry import Registry

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('src_dir', type=str, help='source video directory')
    parser.add_argument('--type', type=str, default="ImageFolder2",help='dataset type')
    parser.add_argument('--labels', nargs="+",type=str,default=[],help='Config file')
    parser.add_argument('--sizes', nargs="+",type=int,default=[],help='statistics by size')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    CLASSIFICATION_REGISTER = Registry("CR")
    CLASSIFICATION_REGISTER.register(ImageFolder)
    CLASSIFICATION_REGISTER.register(ImageFolder2)
    args = parse_args()
    data_dir = args.src_dir
    #dataset_type = args.type
    dataset_type = CLASSIFICATION_REGISTER.get(args.type)
    dataset = dataset_type()
    dataset.read_data(args.src_dir)
    labels_statistics(dataset)