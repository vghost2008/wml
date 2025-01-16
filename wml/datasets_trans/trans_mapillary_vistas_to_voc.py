import sys
import argparse
from wml.iotoolkit.mapillary_vistas_toolkit import *
import numpy as np
import wml.object_detection2.mask as odm
from wml.iotoolkit.pascal_voc_toolkit import write_voc_xml
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
    parser.add_argument("--labels",type=str,nargs="+",help="labels to trans")
    parser.add_argument("--labels-file",type=str,help="labels file to trans")
    args = parser.parse_args()
    return args


def trans_data(data_dir,save_dir,labels):
    wmlu.create_empty_dir(save_dir,remove_if_exists=False)

    label_map = {
        'individual':'person',
        'cyclists':'person',
        'other-rider':'person'
    }

    data = MapillaryVistasData(shuffle=False,
                               allowed_labels_fn=labels,
                               label_map=label_map,
                               use_semantic=False)
    data.read_data(data_dir)

    for i,x in enumerate(data.get_items()):
        full_path, img_info, category_ids, category_names, bboxes, binary_mask, area, is_crowd, num_annotations_skipped = x

        if len(category_names) == 0:
            print(f"Skip {full_path}")
            continue

        base_name = wmlu.base_name(full_path)+".xml"
        xml_path = os.path.join(save_dir,base_name)
        img_path = wmlu.change_dirname(full_path,save_dir)

        if os.path.exists(xml_path):
            print(f"WARNING: File {xml_path} exists.")
        write_voc_xml(xml_path,img_path,img_info,
                      bboxes=bboxes,
                      labels_text=category_names,
                      is_relative_coordinate=False)
        wmlu.try_link(full_path,save_dir)
        sys.stdout.write(f"\r{i}%{len(data)}          ")

def trans_label(label):
    '''
    如果label是一个简单的字符串，直接返回字符串
    如果label是一个name_pattern:new_name, 则如果原数据集中的类型与name_pattern匹配则替换为new_name
    '''
    if ":" in label:
        data = label.split(":")
        data = [x.strip() for x in data]
        if len(data)==2:
            return tuple(data)
        else:
            return label
    return label

def get_labels2trans(args):
    labels = args.labels
    labels_file = args.labels_file
    if labels is None and labels_file is None:
        return None

    if labels_file is not None and isinstance(labels_file,(str,bytes)) and osp.exists(labels_file):
        with open(labels_file,"r") as f:
            lines = f.readlines()
            labels = [x.strip() for x in lines]
            labels = [trans_label(x) for x in labels]
        return labels

    if isinstance(labels,(list,tuple)):
        return labels

    print(f"ERROR labels {labels}, labels file {labels_file}")
    return None

if __name__ == "__main__":
    args = parse_args()
    data_dir = args.src_dir
    save_dir = args.out_dir
    labels = get_labels2trans(args)
    trans_data(data_dir,save_dir,labels)

