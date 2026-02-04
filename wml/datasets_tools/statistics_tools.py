#coding=utf-8
import sys
import os
import numpy as np
from wml.iotoolkit.pascal_voc_toolkit import PascalVOCData
from wml.iotoolkit.coco_toolkit import COCOData
from wml.iotoolkit.labelme_toolkit import LabelMeData
from wml.iotoolkit.fast_labelme import FastLabelMeData
from wml.iotoolkit.mapillary_vistas_toolkit import MapillaryVistasData
from wml.iotoolkit import get_auto_dataset_suffix,check_dataset_dir
from functools import partial
from argparse import ArgumentParser
from wml.iotoolkit.object365v2_toolkit import Object365V2
from wml.object_detection2.data_process_toolkit import remove_class
from wml.iotoolkit.bboxes_statistics import *
import os.path as osp

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('src_dir', type=str, nargs="+",help='source video directory')
    parser.add_argument('--type', type=str, default="auto",help='dataset type')
    parser.add_argument('--labels', nargs="+",type=str,default=[],help='Config file')
    parser.add_argument('--sizes', nargs="+",type=int,default=[],help='statistics by size')
    args = parser.parse_args()
    return args

def read_classes(args):
    try:
        src_dir = osp.abspath(osp.expanduser(args.src_dir[0]))
        file_path = osp.join(src_dir,"classes.txt")
        if not osp.exists(file_path):
            file_path = osp.join(osp.dirname(src_dir),"classes.txt")
            if not osp.exists(file_path):
                print(file_path)
                return []
        with open(file_path,"r") as f:
            lines = f.readlines()
            if len(lines) == 0:
                return None
            data = lines[0].replace(",",";")
            data = data.split(";")
            data = [x.strip() for x in data]
            if len(data)==0:
                return None
            wmlu.print_info(f"Get classes {data} form file {file_path}")
            if len(set(data)) < len(data):
                res = list(set(data))
                wmlu.print_error('Have duplicate class name, classes={data}, auto deduplicate to {res}.')
                return res
            return data
    except:
        return []


if __name__ == "__main__":
    nr = 100
    '''def trans_img_long_size(img_size):
        if img_size[0]<img_size[1]:
            k = 512/img_size[1]
        else:
            k = 512 / img_size[0]
        return [k*img_size[0],k*img_size[1]]
    
    def trans_img_short_size(img_size,min_size=640):
        if img_size[0]<img_size[1]:
            k = min_size/img_size[0]
        else:
            k = min_size/ img_size[1]
        return [k*img_size[0],k*img_size[1]]'''
    args = parse_args()
    data_dir = check_dataset_dir(args.src_dir)

    if args.labels is None or len(args.labels)==0:
        args.labels = read_classes(args)
        if len(args.labels)>0:
            print(f"Auto update labels to {args.labels}")

    dataset_type = args.type
    if dataset_type == "auto":
        dataset_type = get_auto_dataset_suffix(data_dir)

    if dataset_type == "xml":
        dataset = pascal_voc_dataset(data_dir=data_dir,
                                     #labels=args.labels,
                                     )
    elif dataset_type=="json":
        dataset = labelme_dataset(data_dir=data_dir,
                                  #labels=args.labels
                                  )
    elif dataset_type == "coco":
        dataset = coco2017_dataset(data_dir)
                                   #labels=args.labels)
    elif dataset_type == "o365":
        dataset = objects365_dataset(data_dir)
                                   #labels=args.labels)
    elif dataset_type == "vistas":
        dataset = mapillary_vistas_dataset(data_dir)
    else:
        print(f"Unknow dataset type {dataset_type}")
    statics = statistics_boxes_with_datas(dataset,
                                          label_encoder=default_encode_label,
                                          labels_to_remove=None,
                                          max_aspect=None,absolute_size=True,
                                          labels=args.labels,
                                          silent=True)
                                          #trans_img_size=partial(trans_img_long_size_to,long_size=8192))
    statistics_boxes(statics[0], nr=nr)
    statistics_classes_per_img(statics[3])
    statistics_boxes_by_different_area(statics[0],nr=nr,bin_size=5,size_array=args.sizes)
    #statistics_boxes_by_different_ratio(statics[0],nr=nr,bin_size=5)
    #show_boxes_statistics(statics)
    show_classwise_boxes_statistics(statics[1],nr=nr)

    '''data = statics[1]
    boxes = []
    for k,v in data.items():
        if len(k)==1:
            boxes.extend(v)
    data['Char'] = boxes
    show_classwise_boxes_statistics(statics[1],nr=nr,labels=["WD0","WD1","WD2","WD3","Char"])
    '''
