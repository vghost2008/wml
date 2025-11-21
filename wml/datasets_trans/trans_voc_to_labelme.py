import sys
import argparse
from wml.iotoolkit.mapillary_vistas_toolkit import *
import numpy as np
import wml.object_detection2.mask as odm
from wml.iotoolkit.labelme_toolkit import save_labelme_datav3,save_detdata
from wml.iotoolkit.pascal_voc_toolkit import PascalVOCData
import wml.wml_utils as wmlu
import copy
import wml.object_detection2.bboxes as odb
import json
import cv2
import os.path as osp

def parse_args():
    parser = argparse.ArgumentParser(description="build gif")
    parser.add_argument("src_dir",type=str,help="src dir")
    parser.add_argument("out_dir",type=str,help="out dir")
    args = parser.parse_args()
    return args


def trans_data(data_dir,save_dir):
    wmlu.create_empty_dir(save_dir,remove_if_exists=False)


    data = PascalVOCData(label_text2id=None)
    data.read_data(data_dir)

    for i,detdata in enumerate(data.get_items()):
        base_name = wmlu.base_name(detdata.path)+".json"
        save_path = os.path.join(save_dir,base_name)

        if os.path.exists(save_path):
            print(f"WARNING: File {save_path} exists.")
        save_detdata(save_path,detdata.path,detdata,None)
        wmlu.try_link(detdata.path,save_dir)
        sys.stdout.write(f"\r{i}%{len(data)}          ")

def get_labels2trans(args):
    labels = args.labels
    labels_file = args.labels_file
    if labels is None and labels_file is None:
        return None

    if labels_file is not None and isinstance(labels_file,(str,bytes)) and osp.exists(labels_file):
        with open(labels_file,"r") as f:
            lines = f.readlines()
            labels = [x.strip() for x in lines]
        return labels

    if isinstance(labels,(list,tuple)):
        return labels

    print(f"ERROR labels {labels}, labels file {labels_file}")
    return None

if __name__ == "__main__":
    args = parse_args()
    data_dir = args.src_dir
    save_dir = args.out_dir
    trans_data(data_dir,save_dir)

