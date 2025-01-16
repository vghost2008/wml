import wml.img_utils as wmli
import wml.object_detection2.visualization as odv
import matplotlib.pyplot as plt
from wml.iotoolkit.pascal_voc_toolkit import PascalVOCData
from wml.iotoolkit.mapillary_vistas_toolkit import MapillaryVistasData
from wml.iotoolkit.coco_toolkit import COCOData
from wml.iotoolkit.labelme_toolkit import LabelMeData
import argparse
import os.path as osp
import wml.wml_utils as wmlu
import wml.wtorch.utils as wtu
from wml.object_detection2.metrics.toolkit import *
from wml.iotoolkit.image_folder import ImageFolder
import os

def parse_args():
    parser = argparse.ArgumentParser(description='extract optical flows')
    parser.add_argument('dir0', type=str, help='source video directory')
    parser.add_argument('dir1', type=str, help='output rawframe directory')
    parser.add_argument(
        '--ext',
        type=str,
        default='.jpg;;.bmp;;.jpeg;;.png',
        help='video file extensions')
    parser.add_argument('--type', type=str, default='PascalVOCData',help='Data set type')
    args = parser.parse_args()

    return args

def dataset2dict(dataset):
    dataset_dict = wmlu.EDict()
    for label,path in dataset:
        try:
            name = wmlu.base_name(path)
            dataset_dict[name] = label
        except Exception as e:
            print(f"ERROR: {e}")
    return dataset_dict

if __name__ == "__main__":
    args = parse_args()

    dataset0 = ImageFolder()
    dataset0.read_data(args.dir0)

    dataset1 = ImageFolder()
    dataset1.read_data(args.dir1)

    print(f"Process dataset0")
    dataset0_dict = dataset2dict(dataset0)
    print(f"Process dataset1")
    dataset1_dict = dataset2dict(dataset1)

    not_in_list = []
    not_same_label_lst = []

    for k,v in dataset0_dict.items():
        if k not in dataset1_dict:
            not_in_list.append(f"{k} not in dataset1")
        elif v != dataset1_dict[k]:
            not_same_label_lst.append(f"{k} {v} vs {dataset1_dict[k]}")
    for k,v in dataset1_dict.items():
        if k not in dataset0_dict:
            not_in_list.append(f"{k} not in dataset0")
    
    wmlu.show_list(not_in_list)
    wmlu.show_list(not_same_label_lst)


