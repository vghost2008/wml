import argparse
import glob
import os.path as osp
import wml_utils as wmlu
from iotoolkit.pascal_voc_toolkit import write_voc_xml
import numpy as np

'''
给图像文件生成空的xml标注文件
'''
def parse_args():
    parser = argparse.ArgumentParser(
        description='arguments')
    parser.add_argument('img_dir', default=None,type=str,help='img_dir')
    parser.add_argument('--type', default=".xml",type=str,help='img_dir')
    args = parser.parse_args()
    return args

def get_all_imgs(img_dir,img_suffix=".jpg;;.jpeg;;.png;;.bmp"):
    files = wmlu.get_files(img_dir,suffix=img_suffix)
    return files

def make_xml_for_imgs(img_dir,ann_type=".xml"):
    all_img_files = get_all_imgs(img_dir)
    bboxes = np.zeros([0,4])
    labels_text = []
    for file in all_img_files:
        xml_path = wmlu.change_suffix(file,"xml")
        write_voc_xml(xml_path,file,[0,0], bboxes, labels_text)


if __name__ == "__main__":
    args = parse_args()
    make_xml_for_imgs(args.img_dir,ann_type=args.type)