import sys
from wml.iotoolkit.labelme_toolkit import *
import wml.img_utils as wmli
import wml.object_detection2.visualization as odv
from wml.iotoolkit.baidu_mask_toolkit import *
import matplotlib.pyplot as plt
import numpy as np
from wml.object_detection2.basic_datadef import colors_tableau
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
    parser.add_argument("--labels",type=str,nargs="+",help="labels")
    args = parser.parse_args()
    return args


def trans_data(data_dir,save_dir,labels):
    name_to_id_dict = dict(zip(labels,list(range(1,len(labels)+1))))
    wmlu.show_dict(name_to_id_dict)
    wmlu.create_empty_dir(save_dir,remove_if_exists=False)

    def name_to_id(x):
        return name_to_id_dict[x]

    data = LabelMeData(label_text2id=name_to_id, shuffle=False)
    data.read_data(data_dir)
    for i,x in enumerate(data.get_items()):
        full_path, img_info, category_ids, category_names, boxes, binary_mask, area, is_crowd, num_annotations_skipped = x

        if len(category_ids) == 0:
            print(f"Skip {full_path}")
            continue

        new_mask = binary_mask
        r_base_name = wmlu.base_name(full_path)
        #r_base_name = f"IMG_{i+1:05d}"
        base_name = r_base_name+".json"
        save_path = os.path.join(save_dir,base_name)
        img_save_path = os.path.join(save_dir,r_base_name+".jpg")
        wmlu.try_link(full_path,img_save_path)

        new_mask = new_mask.astype(np.uint8)
        labels2name = list(zip(category_ids,category_names))
        BaiDuMaskData.write_json(save_path,new_mask,labels2name,colors_tableau)
        sys.stdout.write(f"\r{i}")

if __name__ == "__main__":
    '''
    Example: python datasets_tools/trans_labelme_to_baidu.py ~/下载/EISeg/images/ ~/a --labels burnt puncture crease scratch
    '''
    args = parse_args()
    data_dir = args.src_dir
    save_dir = args.out_dir
    labels = args.labels
    trans_data(data_dir, save_dir,labels)
