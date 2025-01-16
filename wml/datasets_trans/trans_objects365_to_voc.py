import sys
import argparse
from wml.iotoolkit.object365v2_toolkit import Object365V2
import numpy as np
import wml.object_detection2.mask as odm
from wml.iotoolkit.pascal_voc_toolkit import write_voc_xml
import wml.wml_utils as wmlu
import copy
import wml.object_detection2.bboxes as odb
import json
import cv2
import os
import os.path as osp

def parse_args():
    parser = argparse.ArgumentParser(description="build gif")
    parser.add_argument("ann_path",type=str,help="ann path")
    parser.add_argument("out_dir",type=str,help="out dir")
    parser.add_argument("--no-crowd",action="store_true",help="whether to use files contains crowd objects")
    args = parser.parse_args()
    return args


def trans_data(ann_path,save_dir,no_crowd=True):
    wmlu.create_empty_dir(save_dir,remove_if_exists=False)

    data = Object365V2(is_relative_coordinate=False,remove_crowd=False)
    data.read_data(ann_path)

    for i,x in enumerate(data.get_items()):
        full_path, img_info, category_ids, category_names, bboxes, binary_mask, area, is_crowd, num_annotations_skipped = x

        if len(category_names) == 0:
            print(f"Skip {full_path}")
            continue
        if no_crowd and np.any(is_crowd):
            continue

        dir_name = wmlu.base_name(osp.dirname(full_path))
        t_save_dir = osp.join(save_dir,dir_name)
        os.makedirs(t_save_dir,exist_ok=True)

        base_name = wmlu.base_name(full_path)+".xml"
        xml_path = os.path.join(t_save_dir,base_name)
        img_path = wmlu.change_dirname(full_path,t_save_dir)

        if os.path.exists(xml_path):
            print(f"WARNING: File {xml_path} exists.")
        write_voc_xml(xml_path,img_path,img_info,
                      bboxes=bboxes,
                      labels_text=category_names,
                      is_relative_coordinate=False,
                      difficult=is_crowd)
        wmlu.try_link(full_path,t_save_dir)
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
    ann_path = args.ann_path
    save_dir = args.out_dir
    no_crowd = True   #args.no_crowd
    #labels = get_labels2trans(args)
    trans_data(ann_path,save_dir,no_crowd=no_crowd)

