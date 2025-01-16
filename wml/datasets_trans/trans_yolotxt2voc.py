import sys
import os
import wml.wml_utils as wmlu
import cv2
import os.path as osp
import numpy as np
from wml.iotoolkit.pascal_voc_toolkit import write_voc_xml
import argparse
import wml.img_utils as wmli

def parse_args():
    parser = argparse.ArgumentParser(description="build gif")
    parser.add_argument("src_dir",type=str,help="src dir")
    parser.add_argument("--labels",type=str,nargs="+",default=[],help="labels")
    args = parser.parse_args()
    return args

'''
txt 文件内容
label cx cy w h (相对坐标), label>=0
Example:
9 0.585337 0.310909 0.007212 0.056970
9 0.655950 0.309697 0.010216 0.049697
9 0.695913 0.295758 0.008413 0.041212
1 0.421575 0.299394 0.022236 0.038788
1 0.380409 0.298788 0.024038 0.030303
9 0.668570 0.289091 0.007812 0.040000
1 0.890925 0.656364 0.213341 0.372121
1 0.452825 0.299394 0.019832 0.041212
9 0.814002 0.377576 0.022236 0.105455
...

'''

default_classes_names = {}
for i in range(1000):
    default_classes_names[i] = str(i)

def read_yolotxt(txt_path,img_suffix="jpg"):
    labels = []
    bboxes = []
    with open(txt_path,"r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line = line.split(" ")
            labels.append(int(line[0]))
            cx = float(line[1])
            cy = float(line[2])
            w = float(line[3])
            h = float(line[4])
            xmin = cx-w/2
            ymin = cy-h/2
            xmax = cx+w/2
            ymax = cy+h/2
            bboxes.append([ymin,xmin,ymax,xmax])
    return np.array(labels),np.array(bboxes)


def trans_yolotxt(txt_path,img_path,classes_names):
    if not osp.exists(txt_path):
        print(f"{txt_path} not exists.")
        return
    labels,bboxes = read_yolotxt(txt_path)
    xml_path = wmlu.change_suffix(txt_path,"xml")
    _bboxes = []
    _labels = []
    for i,x in enumerate(labels):
        if x not in classes_names:
            continue
        _labels.append(classes_names[x])
        _bboxes.append(bboxes[i])
    if len(_labels) == 0:
        print(f"{txt_path} is empty.")
    print(f"Save {xml_path}")
    write_voc_xml(xml_path,img_path,None,_bboxes,_labels)

def trans_dirs(dir_path,classes_names):
    img_files = wmlu.get_files(dir_path,suffix=wmli.BASE_IMG_SUFFIX)
    for img_path in img_files:
        txt_path = wmlu.change_suffix(img_path,"txt")
        trans_yolotxt(txt_path,img_path,classes_names)

def read_labels_file(path):
    with open(path,"r") as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
    return lines

if __name__ == "__main__":
    #classes_names = ["car", "truck", "tank_truck", "bus", "van", "dangerous_sign"]
    args = parse_args()
    src_dir = args.src_dir
    labels = args.labels
    if len(labels)==0:
        labels_file = os.path.join(src_dir,"classes.txt")
        if not os.path.exists(labels_file):
            print(f"Find labels file {labels_file} faild.")
            exit(0)
        labels = read_labels_file(labels_file)
    id2text = dict(zip(range(len(labels)),labels))
    wmlu.show_dict(id2text)
        
    trans_dirs(src_dir,id2text)

