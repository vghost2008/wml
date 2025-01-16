import sys
from wml.iotoolkit.labelme_toolkit import *
import wml.img_utils as wmli
from wml.iotoolkit.baidu_mask_toolkit import *
import matplotlib.pyplot as plt
import numpy as np
from wml.iotoolkit.labelme_toolkit_fwd import get_files as get_labelme_files
from wml.iotoolkit.labelme_toolkit_fwd import read_labelme_mckp_data
import wml.wml_utils as wmlu
import argparse
import pickle
from wml.object_detection2.standard_names import *

def parse_args():
    parser = argparse.ArgumentParser(description="build gif")
    parser.add_argument("src_dir",type=str,help="src dir")
    parser.add_argument("out_dir",type=str,help="out dir")
    parser.add_argument("--labels",type=str,nargs="+",help="labels")
    parser.add_argument("--dont-save-imgs", default=False, action="store_true", help="save image data.")
    args = parser.parse_args()
    return args


def trans_data(data_dir,save_dir,labels,args):
    wmlu.create_empty_dir(save_dir,remove_if_exists=False)


    files = get_labelme_files(data_dir)
    for i,(img_file,json_file) in enumerate(files):
        image_info,labels,points = read_labelme_mckp_data(json_file)
        datas = {}
        datas[IMG_INFO] = image_info
        datas[GT_LABELS] = labels
        datas[GT_KEYPOINTS] = points
        if not args.dont_save_imgs:
            img_data = wmli.encode_img(img_file)
            datas[IMAGE] = img_data
        rel_path = wmlu.get_relative_path(json_file,data_dir)
        save_path = osp.join(save_dir,rel_path)
        save_path = wmlu.change_suffix(save_path,"bin")
        with open(save_path,"wb") as f:
            pickle.dump(datas,f)
        sys.stdout.write(f"\r{i}")

if __name__ == "__main__":
    '''
    '''
    args = parse_args()
    data_dir = args.src_dir
    save_dir = args.out_dir
    labels = args.labels
    trans_data(data_dir, save_dir,labels,args)
