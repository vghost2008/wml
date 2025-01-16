#coding=utf-8
import sys
import os
import wml.object_detection2.npod_toolkit as npod
import wml.wml_utils
import matplotlib.pyplot as plt
import numpy as np
import math
import wml.object_detection2.visualization as odv
import wml.img_utils as wmli
from wml.iotoolkit.pascal_voc_toolkit import PascalVOCData,read_voc_xml
from wml.iotoolkit.coco_toolkit import COCOData
from wml.iotoolkit import FastLabelMeData
import wml.object_detection2.bboxes as odb 
import pandas as pd
import wml.wml_utils as wmlu
from wml.iotoolkit.mapillary_vistas_toolkit import MapillaryVistasData
from wml.iotoolkit import get_auto_dataset_suffix
from sklearn.cluster import KMeans
from functools import partial
from argparse import ArgumentParser
from itertools import count
import os.path as osp
import wml.img_utils as wmli

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('data_dir', type=str, help='source video directory')
    parser.add_argument('save_dir', type=str, help='save dir')
    parser.add_argument('--type', type=str, default="auto",help='dataset type')
    parser.add_argument('--labels', nargs="+",type=str,default=[],help='Config file')
    parser.add_argument('--min-size', type=int, default=0,help='min bbox size')
    parser.add_argument('--add-classes-name', action='store_true', help='min bbox size')
    parser.add_argument('-np','--name-prefix', type=str, help='dataset type')
    parser.add_argument('--no-draw',action='store_true', help="don't draw bbox on img.")
    args = parser.parse_args()
    return args

def test_dataset():
    data = PascalVOCData(label_text2id=None)
    data.read_data("/home/vghost/ai/mldata2/test_data_0day/test_s")

    return data.get_items()

def pascal_voc_dataset(data_dir,labels=None):
    #labels = ['MS7U', 'MP1U', 'MU2U', 'ML9U', 'MV1U', 'ML3U', 'MS1U', 'Other']
    if labels is not None and len(labels)>0:
        label_text2id = dict(zip(labels,count()))
    else:
        label_text2id = None
    
    #data = PascalVOCData(label_text2id=label_text2id,resample_parameters={6:8,5:2,7:2})
    data = PascalVOCData(label_text2id=label_text2id,absolute_coord=True)

    '''data_path = "/mnt/data1/wj/ai/smldata/boedcvehicle/train"
    data_path = "/mnt/data1/wj/ai/smldata/boedcvehicle/wt_06"
    data_path = "/home/wj/ai/mldata1/GDS1Crack/train"
    data_path = "/home/wj/ai/mldata1/take_photo/train/coco"
    data_path = "/mnt/data1/wj/ai/mldata/MOT/MOT17/train/MOT17-09-SDP/img1"
    data_path = "/home/wj/ai/mldata1/B11ACT/datas/labeled"
    data_path = "/home/wj/ai/mldata1/B7mura/datas/data/ML3U"
    data_path = "/home/wj/ai/mldata1/B7mura/datas/data/MV1U"
    data_path = "/home/wj/ai/mldata1/B7mura/datas/data/MU4U"
    data_path = "/home/wj/ai/mldata1/B7mura/datas/data"
    data_path = "/home/wj/下载/_数据集"'''
    #data_path = "/home/wj/ai/mldata1/B7mura/datas/test_s0"
    #data_path = "/home/wj/0day/wt_06"
    #data_path = '/home/wj/0day/pyz'
    data.read_data(data_dir,
                   silent=True,
                   img_suffix=".bmp;;.jpg")

    return data

def coco2014_dataset():
    data = COCOData()
    data.read_data(wmlu.home_dir("ai/mldata/coco/annotations/instances_train2014.json"), 
                   image_dir=wmlu.home_dir("ai/mldata/coco/train2014"))

    return data.get_items()

def coco2017_dataset():
    data = COCOData()
    data.read_data(wmlu.home_dir("ai/mldata2/coco/annotations/instances_train2017.json"),
                   image_dir=wmlu.home_dir("ai/mldata2/coco/train2017"))

    return data.get_items()

def coco2014_val_dataset():
    data = COCOData()
    data.read_data(wmlu.home_dir("ai/mldata/coco/annotations/instances_val2014.json"),
                   image_dir=wmlu.home_dir("ai/mldata/coco/val2014"))

    return data.get_items()

def labelme_dataset(data_dir,labels):
    data = FastLabelMeData(label_text2id=None,absolute_coord=True)
    #data.read_data("/home/vghost/ai/mldata2/qualitycontrol/rdatasv10")
    data.read_data(data_dir,img_suffix="bmp;;jpg;;jpeg")
    #data.read_data("/home/wj/ai/mldata1/B11ACT/datas/test_s0",img_suffix="bmp")
    return data


lid = 0
def mapillary_vistas_dataset():
    NAME2ID = {}
    ID2NAME = {}

    def name_to_id(x):
        global lid
        if x in NAME2ID:
            return NAME2ID[x]
        else:
            NAME2ID[x] = lid
            ID2NAME[lid] = x
            lid += 1
            return NAME2ID[x]
    ignored_labels = [
        'manhole', 'dashed', 'other-marking', 'static', 'front', 'back',
        'solid', 'catch-basin','utility-pole', 'pole', 'street-light','direction-back', 'direction-front'
         'ambiguous', 'other','text','diagonal','left','right','water-valve','general-single','temporary-front',
        'wheeled-slow','parking-meter','split-left-or-straight','split-right-or-straight','zigzag',
        'give-way-row','ground-animal','phone-booth','give-way-single','garage','temporary-back','caravan','other-barrier'
    ]
    data = MapillaryVistasData(label_text2id=name_to_id, shuffle=False, ignored_labels=ignored_labels)
    # data.read_data("/data/mldata/qualitycontrol/rdatasv5_splited/rdatasv5")
    # data.read_data("/home/vghost/ai/mldata2/qualitycontrol/rdatav10_preproc")
    # data.read_data("/home/vghost/ai/mldata2/qualitycontrol/rdatasv10_neg_preproc")
    data.read_data(wmlu.home_dir("ai/mldata/mapillary_vistas/mapillary-vistas-dataset_public_v2.0"))
    return data.get_boxes_items()

def cut_and_save_imgs_in_bboxes(dataset,save_dir,min_size=0,add_classes_name=False,args=None):
    counter = wmlu.Counter()
    for idx,data in enumerate(dataset):
        try:
            img_file, shape,labels, labels_names, bboxes,*_ = data
            if len(labels_names)==0:
                continue
            sys.stdout.write(f"Process {idx}/{len(dataset)}\r")
            org_bboxes = bboxes.copy()
            bboxes = odb.npscale_bboxes(bboxes,1.1)
            if min_size>1:
                bboxes = odb.clamp_bboxes(bboxes,min_size=min_size)
            bboxes = bboxes.astype(np.int32)
            img = wmli.imread(img_file)
            base_name = wmlu.base_name(img_file)
            for i,name in enumerate(labels_names):
                if args.name_prefix is not None:
                    name = args.name_prefix+name
                v = counter.add(name)
                if add_classes_name:
                    t_save_path = osp.join(save_dir,f"{name}_{base_name}.jpg")
                else:
                    t_save_path = osp.join(save_dir,name,f"{base_name}.jpg")
                t_save_path = wmlu.get_unused_path_with_suffix(t_save_path,v)
                bbox = bboxes[i]
                simg = wmli.crop_img_absolute(img,bbox)
                if min_size>1:
                    offset = np.array([bbox[0],bbox[1],bbox[0],bbox[1]])
                    if not args.no_draw:
                        simg = odv.draw_bbox(simg,org_bboxes[i]-offset,thickness=1,xy_order=False)
                wmli.imwrite(t_save_path,simg)
        except Exception as e:
            print(f"ERROR: {e}")

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
    data_dir = args.data_dir
    dataset_type = args.type
    if dataset_type == "auto":
        dataset_type = get_auto_dataset_suffix(data_dir)
    if dataset_type == "xml":
        dataset = pascal_voc_dataset(data_dir=data_dir,
                                     labels=args.labels,
                                     )
    elif dataset_type=="json":
        dataset = labelme_dataset(data_dir=data_dir,
                                  labels=args.labels
                                  )
    
    cut_and_save_imgs_in_bboxes(dataset,args.save_dir,min_size=args.min_size,add_classes_name=args.add_classes_name,args=args)
    
