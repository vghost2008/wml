import wml.img_utils as wmli
import wml.object_detection2.visualization as odv
import matplotlib.pyplot as plt
from wml.iotoolkit.pascal_voc_toolkit import PascalVOCData
from wml.iotoolkit.mapillary_vistas_toolkit import MapillaryVistasData
from wml.iotoolkit.coco_toolkit import COCOData
from wml.iotoolkit.labelme_toolkit import LabelMeData,save_detdata
from wml.iotoolkit.fast_labelme import FastLabelMeData
from wml.iotoolkit.labelmemlines_dataset import LabelmeMLinesDataset
from wml.wstructures.detdata import resize_detdata
import argparse
import os.path as osp
import os
import wml.wml_utils as wmlu
import wml.wtorch.utils as wtu
from wml.iotoolkit import get_auto_dataset_type
import numpy as np
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description='extract optical flows')
    parser.add_argument('src_dir', type=str, help='source video directory')
    parser.add_argument('out_dir', type=str, help='output rawframe directory')
    parser.add_argument(
        '-S','--size', type=int, nargs="+", default=None,help='resize image [w,h]')
    parser.add_argument(
        '-s','--scale', type=float, nargs="+", default=None,help='resize image [w,h]')
    parser.add_argument(
        '--ext',
        type=str,
        default=wmli.BASE_IMG_SUFFIX,
        #choices=['avi', 'mp4', 'webm','MOV'],
        help='video file extensions')
    parser.add_argument(
        '--new-height', type=int, default=0, help='resize image height')
    parser.add_argument('--type', type=str, default='auto',help='Data set type')
    parser.add_argument(
        '--line-width', type=int, default=2, help='line width')
    parser.add_argument(
        '--view-nr', type=int, default=-1, help='view dataset nr.')
    parser.add_argument('--suffix', type=str, default="_view",help='suffix to output')
    parser.add_argument(
        '-ci',
        '--copy-imgs',
        action='store_true',
        help='whether copy raw img to target')
    parser.add_argument(
        '--base-name',
        '-bn',
        action='store_true',
        help='save file with base name.')
    parser.add_argument(
        '--no-text',
        '-nt',
        action='store_true',
        help='no label name.')
    parser.add_argument(
        '-cn',
        '--channel-names', type=str, nargs="+", help='image channel names for mci image.')
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
    view_nr = args.view_nr
    shuffle = view_nr>0
    if args.type == "auto":
        dataset_type = get_auto_dataset_type(args.src_dir)
    else:
        print(DATASETS,args.type)
        dataset_type = DATASETS[args.type]
    print(dataset_type)
    data = dataset_type(label_text2id=None,shuffle=shuffle,absolute_coord=True)
    data.read_data(args.src_dir,img_suffix=args.ext)

    if args.no_text:
        text_fn = no_text_fn
    else:
        text_fn = normal_text_fn

    if view_nr>0:
        data.files = data.files[:view_nr]


    new_size = args.size
    scale = args.scale
    for x in data.get_items():
        full_path, img_info,category_ids, category_names, boxes,binary_masks,area,is_crowd,*_ =  x

        old_size = wmli.get_img_size(full_path)[::-1]
        if new_size is None:
            new_size = [int(scale[0]*old_size[0]),int(scale[1]*old_size[1])]

        detdata = resize_detdata(x,new_size,old_size=old_size)

        print(full_path)
        if args.base_name:
            filename = osp.basename(full_path)
        else:
            filename = wmlu.get_relative_path(full_path,args.src_dir)

        raw_save_path = osp.join(args.out_dir,filename)
        wmlu.make_dir_for_file(raw_save_path)
        img = wmli.imread(full_path)
        img = wmli.resize_img(img,new_size)
        wmli.imwrite(raw_save_path,img)
        ann_save_path = wmlu.change_suffix(raw_save_path,"json")
        save_detdata(ann_save_path,raw_save_path,detdata)


    print(f"Save dir: {args.out_dir}")
