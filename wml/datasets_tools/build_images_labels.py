import numpy as np
import cv2 as cv
import imageio
import glob
import os
import sys
import os.path as osp
import wml.img_utils as wmli
import wml.wml_utils as wmlu
import argparse
import shutil
from wml.iotoolkit.labelme_toolkit_fwd import get_files,read_labelme_data


def parse_args():
    parser = argparse.ArgumentParser(description="build gif")
    parser.add_argument("src_dir",type=str,help="src dir")
    parser.add_argument("out_dir",type=str,help="out dir")
    parser.add_argument("--labels",type=str,nargs="+",default=None,help="video ext")
    args = parser.parse_args()
    return args

def get_labels(files):
    res = set()
    for imgf,annf in files:
        data = read_labelme_data(annf,mask_on=False,return_type=1)
        labels = data.labels_name
        res = res|set(list(labels))
    return list(res)


if __name__ == "__main__":
    args = parse_args()
    wmlu.create_empty_dir(args.out_dir,remove_if_exists=False)
    files = get_files(args.src_dir)
    img_save_dir = osp.join(args.out_dir,"images")
    json_save_dir = osp.join(args.out_dir,"labels")
    classes_save_path = osp.join(args.out_dir,"classes.txt")
    for imgf,annf in files:
        bn = wmlu.get_relative_path(imgf,args.src_dir)
        save_path = osp.join(img_save_dir,bn)
        dir_path = osp.dirname(save_path)
        os.makedirs(dir_path,exist_ok=True)
        shutil.copy(imgf,save_path)

        bn = wmlu.get_relative_path(annf,args.src_dir)
        save_path = osp.join(json_save_dir,bn)
        dir_path = osp.dirname(save_path)
        os.makedirs(dir_path,exist_ok=True)
        shutil.copy(annf,save_path)

    labels = args.labels
    if labels is None or len(labels) == 0:
        labels = get_labels(files)
    if len(labels)>0:
        info = labels[0]
        for l in labels[1:]:
            info += f"; {l}"
        with open(classes_save_path,"w") as f:
            f.write(info)


