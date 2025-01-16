import sys
import argparse
from wml.iotoolkit.mapillary_vistas_toolkit import *
import numpy as np
import wml.object_detection2.mask as odm
from wml.iotoolkit.labelme_toolkit import save_labelme_datav3
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


    data = MapillaryVistasData(shuffle=False,
                                allowed_labels_fn=labels,
                               label_map=None)
    data.read_data(data_dir)

    for i,x in enumerate(data.get_items()):
        full_path, img_info, category_ids, category_names, bboxes, binary_mask, area, is_crowd, num_annotations_skipped = x

        if len(category_names) == 0:
            print(f"Skip {full_path}")
            continue

        bboxes = odb.npchangexyorder(bboxes) #to [x0,y0,x1,y1] fmt
        new_mask = odm.crop_masks_by_bboxes(binary_mask,bboxes)
        base_name = wmlu.base_name(full_path)+".json"
        save_path = os.path.join(save_dir,base_name)

        if os.path.exists(save_path):
            print(f"WARNING: File {save_path} exists.")
        img_shape = dict(height=img_info[0],width=img_info[1])
        save_labelme_datav3(save_path,
                            full_path,
                            img_shape,
                            labels=category_names,
                            bboxes=bboxes,
                            masks=new_mask,
                            label_to_text=None,
                            )
        wmlu.try_link(full_path,save_dir)
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
    labels = get_labels2trans(args)
    trans_data(data_dir,save_dir,labels)

