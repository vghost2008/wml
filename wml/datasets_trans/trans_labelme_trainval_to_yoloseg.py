import sys
from wml.iotoolkit import FastLabelMeData
import wml.img_utils as wmli
import wml.object_detection2.visualization as odv
from wml.iotoolkit.baidu_mask_toolkit import *
import matplotlib.pyplot as plt
import numpy as np
from wml.iotoolkit.yolo_toolkit import write_yoloseg_txt
import wml.object_detection2.mask as odm
import wml.wml_utils as wmlu
import copy
import json
import cv2
import shutil
import argparse
from datasets_trans.trans_labelme_to_yoloseg import *

def parse_args():
    parser = argparse.ArgumentParser(description="build gif")
    parser.add_argument("src_dir",type=str,help="src dir")
    parser.add_argument("out_dir",type=str,help="out dir")
    parser.add_argument("--labels",type=str,nargs="+",help="labels")
    parser.add_argument("--labels-dict",type=str,help="labels")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    '''
    Example: 
    python datasets_trans/trans_labelme_trainval_to_yoloseg.py ~/ai/mldata1/B10CFOD/ito_datasets/datasetv1.0/  ~/ai/mldata1/B10CFOD/ito_datasets/yolo_datasetv1.0/ --labels 'BW' 'HQ' 'SC' 'YW' 'FB'
    '''
    args = parse_args()
    data_dir = args.src_dir
    save_dir = args.out_dir
    labels = args.labels
    sub_set = "train"
    trans_data(osp.join(data_dir,sub_set), save_dir,labels,sub_set=sub_set,args=args)
    sub_set = "val"
    trans_data(osp.join(data_dir,sub_set), save_dir,labels,sub_set=sub_set,args=args)
