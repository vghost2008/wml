import wml.img_utils as wmli
import wml.object_detection2.visualization as odv
import matplotlib.pyplot as plt
from wml.iotoolkit.pascal_voc_toolkit import PascalVOCData
from wml.iotoolkit.mapillary_vistas_toolkit import MapillaryVistasData
from wml.iotoolkit.coco_toolkit import COCOData
from wml.iotoolkit.labelme_toolkit import LabelMeData
from wml.iotoolkit.fast_labelme import FastLabelMeData
import argparse
import os.path as osp
import os
import wml.wml_utils as wmlu
import wml.wtorch.utils as wtu
from wml.iotoolkit import get_auto_dataset_type
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description='extract optical flows')
    parser.add_argument('src_dir', type=str, help='source video directory')
    parser.add_argument('--out-dir', type=str, help='output rawframe directory')
    return parser.parse_args()

def get_text(file_path):
    return wmlu.base_name(file_path)[-8:]

def draw_text_on_img(img_path,src_dir,save_dir):
    img = wmli.imread(img_path)
    text = get_text(img_path)
    img = odv.draw_text_on_image(img,text,font_scale=1.0,pos="tl")
    if save_dir is None:
        save_path = img_path
    else:
        rname = wmlu.get_relative_path(img_path,src_dir)
        save_path = osp.join(save_dir,rname)
        cur_save_dir = osp.dirname(save_path)
        if not osp.exists(cur_save_dir):
            os.makedirs(cur_save_dir)
    
    wmli.imwrite(save_path,img)

if __name__ == "__main__":
    args = parse_args()
    files = wmlu.get_files(args.src_dir,suffix=wmli.BASE_IMG_SUFFIX)
    for file in files:
        draw_text_on_img(file,args.src_dir,args.out_dir)