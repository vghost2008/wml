import wml.img_utils as wmli
import wml.object_detection2.visualization as odv
import matplotlib.pyplot as plt
from wml.iotoolkit.pascal_voc_toolkit import PascalVOCData
from wml.iotoolkit.mapillary_vistas_toolkit import MapillaryVistasData
from wml.iotoolkit.coco_toolkit import COCOData
from wml.iotoolkit.labelme_toolkit import LabelMeData
from wml.iotoolkit.fast_labelme import FastLabelMeData
from wml.iotoolkit.labelme_toolkit_fwd import save_detdata
from wml.iotoolkit.labelmemlines_dataset import LabelmeMLinesDataset
import argparse
import os.path as osp
import os
import wml.wml_utils as wmlu
import wml.wtorch.utils as wtu
from wml.iotoolkit import get_auto_dataset_type
import numpy as np
import copy
import shutil

'''
将标注文件中的文件名修改为label-name, 如果没有设置label-name,则修改为目录名
'''

def parse_args():
    parser = argparse.ArgumentParser(description='extract optical flows')
    parser.add_argument('src_dir', type=str, help='source video directory')
    parser.add_argument('out_dir', type=str, help='output rawframe directory')
    parser.add_argument('--label-name', type=str, help='change label name')
    parser.add_argument(
        '--ext',
        type=str,
        default=wmli.BASE_IMG_SUFFIX,
        #choices=['avi', 'mp4', 'webm','MOV'],
        help='video file extensions')
    parser.add_argument('--type', type=str, default='auto',help='Data set type')
    parser.add_argument(
        '--copy-imgs',
        action='store_true',
        help='whether copy raw img to target')
    args = parser.parse_args()

    return args

def normal_text_fn(x,scores):
    return x

def no_text_fn(x,scores):
    return ""

DATASETS = {}

def register_dataset(type):
    DATASETS[type.__name__] = type

register_dataset(PascalVOCData)
register_dataset(COCOData)
register_dataset(MapillaryVistasData)
register_dataset(LabelMeData)
register_dataset(FastLabelMeData)
register_dataset(LabelmeMLinesDataset)

def simple_names(x):
    if "--" in x:
        return x.split("--")[-1]
    return x

if __name__ == "__main__":

    args = parse_args()
    if args.type == "auto":
        dataset_type = get_auto_dataset_type(args.src_dir)
    else:
        print(DATASETS,args.type)
        dataset_type = DATASETS[args.type]
    data = dataset_type(label_text2id=None,shuffle=False,absolute_coord=True)
    data.read_data(args.src_dir,img_suffix=args.ext)



    if args.copy_imgs:
        if args.suffix is None or len(args.suffix)==0:
            args.suffix = "_view"

    label_name = args.label_name

    for x in data.get_items():
        full_path, img_info,category_ids, category_names, boxes,binary_masks,area,is_crowd,*_ =  x
        print(full_path,dataset_type)
        category_names = [simple_names(x) for x in category_names]
        img = wmli.imread(full_path)
        old_shape = img.shape

        filename = wmlu.get_relative_path(full_path,args.src_dir)
        save_path = osp.join(args.out_dir,filename)
        wmlu.make_dir_for_file(save_path)
        cur_labels = copy.deepcopy(x.labels_name)
        if label_name is not None and len(label_name)>0:
            new_label_name = label_name
        else:
            dirname = osp.dirname(full_path)
            new_label_name = wmlu.base_name(dirname)
        new_labels = [new_label_name]*len(cur_labels)
        x = x._replace(labels_name=new_labels)
        shutil.copy(full_path,save_path)
        if isinstance(data,(LabelMeData,FastLabelMeData)):
            print(f"Change {filename} labels from {cur_labels} to {new_labels}")
            json_path = wmlu.change_suffix(save_path,"json")
            save_detdata(json_path,save_path,x)



    print(f"Save dir: {args.out_dir}")
