import sys
from wml.iotoolkit.labelme_toolkit import *
import wml.img_utils as wmli
import wml.object_detection2.visualization as odv
from wml.iotoolkit.baidu_mask_toolkit import *
import matplotlib.pyplot as plt
import numpy as np
from wml.iotoolkit.yolo_toolkit import write_yolo_txt
import wml.object_detection2.mask as odm
import wml.wml_utils as wmlu
import copy
import json
import cv2
import shutil
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="build gif")
    parser.add_argument("src_dir",type=str,help="src dir")
    parser.add_argument("out_dir",type=str,help="out dir")
    parser.add_argument("--sub-set",type=str,help="labels")
    parser.add_argument("--labels",type=str,nargs="+",help="labels")
    parser.add_argument("--labels-dict",type=str,help="labels")
    args = parser.parse_args()
    return args

def write_labels(save_dir,labels):
    os.makedirs(save_dir,exist_ok=True)
    save_path = osp.join(save_dir,"classes.txt")
    with open(save_path,"w") as f:
        for i,l in enumerate(labels):
            f.write(f"    {i}: {l}\n")

def trans_data(data_dir,save_dir,labels,sub_set,args):
    write_labels(save_dir,labels)
    name_to_id_dict = dict(zip(labels,list(range(len(labels)))))
    if args.labels_dict is not None:
        tmp_dict = eval(args.labels_dict)
        name_to_id_dict.update(tmp_dict)
    wmlu.show_dict(name_to_id_dict)
    wmlu.create_empty_dir(save_dir,remove_if_exists=False)

    def name_to_id(x):
        return name_to_id_dict[x]

    data = LabelMeData(label_text2id=name_to_id, shuffle=False)
    data.read_data(data_dir)
    imgs_save_dir = osp.join(save_dir,"images")
    labels_save_dir = osp.join(save_dir,"labels")
    if sub_set is not None:
        imgs_save_dir = osp.join(imgs_save_dir,sub_set)
        labels_save_dir = osp.join(labels_save_dir,sub_set)
    os.makedirs(imgs_save_dir,exist_ok=True)
    os.makedirs(labels_save_dir,exist_ok=True)
    
    for i,x in enumerate(data.get_items()):
        full_path, img_info, category_ids, category_names, boxes, binary_mask, area, is_crowd, num_annotations_skipped = x

        if len(category_ids) == 0:
            print(f"Skip {full_path}")
            continue

        img_save_path = osp.join(imgs_save_dir,osp.basename(full_path))
        shutil.copy(full_path,img_save_path)

        txt_save_path = osp.join(labels_save_dir,wmlu.base_name(full_path)+".txt")
        write_yolo_txt(txt_save_path,img_save_path,category_ids,boxes)

        sys.stdout.write(f"\r{i}/{len(data)}")

if __name__ == "__main__":
    '''
    Example: 
    python datasets_trans/trans_labelme_to_yolo.py ~/ai/mldata1/B10AOIOLD/datasets/datasetv4.0/val/ ~/ai/mldata1/B10AOIOLD/datasets/yolo_datasetv4.0/ --labels 'LD' 'AD' 'QP' 'LM' 'Gap' 'DM' --labels-dict "{ 'LD' : 0 , 'AD' : 1 , 'QP' : 2 , 'LM' : 3 , 'Gap' : 4 , 'DM' : 5 , 'XYBX_LM' : 3 , 'XB_LM' : 3 , 'HVBX_LM' : 3 , 'ETC_LM' : 3 , 'PG_G' : 4 , 'BG_G' : 4 , 'BD_G' : 4 , 'HD_DM' : 1 , 'HM_DM' : 5 , 'HG_DM' : 5 , 'CM_DM' : 5 , }" --sub-set val
    '''
    args = parse_args()
    data_dir = args.src_dir
    save_dir = args.out_dir
    labels = args.labels
    trans_data(data_dir, save_dir,labels,sub_set=args.sub_set,args=args)
