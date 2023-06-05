#coding=utf-8
import sys
import os
import object_detection2.npod_toolkit as npod
import wml_utils
import numpy as np
import math
import object_detection2.visualization as odv
import img_utils as wmli
from iotoolkit.pascal_voc_toolkit import PascalVOCData,read_voc_xml,write_voc_xml
from iotoolkit.coco_toolkit import COCOData
from iotoolkit.labelme_toolkit import LabelMeData
import object_detection2.bboxes as odb 
import wml_utils as wmlu
from iotoolkit.mapillary_vistas_toolkit import MapillaryVistasData
from functools import partial
from argparse import ArgumentParser
from itertools import count
import os.path as osp
import img_utils as wmli
import shutil

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('patch_dir', type=str, help='patch annotation directory')
    parser.add_argument('data_dir', type=str, help='data dir')
    parser.add_argument('--type', type=str, default="xml",help='dataset type')
    args = parser.parse_args()
    return args

def patch_one_xml_file(bbox,label,xml_path,img_path):
    print(f"Patch {xml_path}")
    shape, bboxes, labels_text, difficult, truncated, probs = read_voc_xml(xml_path,absolute_coord=True)
    bk_xml_path = wmlu.change_suffix(xml_path,"bk")
    shutil.move(xml_path,bk_xml_path)
    bbox = odb.npchangexyorder([bbox])
    bboxes = np.concatenate([bboxes,bbox],axis=0)
    labels_text.append(label)
    write_voc_xml(xml_path,img_path,shape, bboxes, labels_text, is_relative_coordinate=False)
    
def get_patch_info(patch_path):
    '''
    patch_patch: [org_file_name]_bbox{xmin},{ymin},{xmax},{ymax}_{label}.jpg
    Example:
    765K230022C1A_T380A0N_SCANIMAGE_20220408_174713_494_bbox1737,551,1753,562_MP1U.jpg
    '''
    patch_path = osp.basename(patch_path)
    keyword = "_bbox"
    idx = patch_path.rfind(keyword)
    if idx<0:
        return None
    basename = patch_path[:idx]
    info = patch_path[idx+len(keyword):]
    idx1 = info.rfind(".")
    info = info[:idx1]
    info = info.split("_")
    if len(info)<2:
        print(f"ERROR: error info format {patch_path}, {info}")
        return None
    label = info[1]
    bbox = info[0].split(",")
    if len(bbox)<4:
        print(f"ERROR: error info format {patch_path}, {info}, bbox={bbox}")
        return None
    bbox = [float(x) for x in bbox]
    '''
    bbox: xmin,ymin,xmax,ymax
    '''
    return basename,bbox,label
    

if __name__ == "__main__":
    args = parse_args()
    patch_dir = args.patch_dir
    data_dir = args.data_dir
    suffix = args.type
    patch_files = wmlu.get_files(patch_dir,suffix=wmli.BASE_IMG_SUFFIX)
    data_files = wmlu.get_files(data_dir,suffix=wmli.BASE_IMG_SUFFIX)
    data_dict = wmlu.EDict()
    for f in data_files:
        try:
            basename = wmlu.base_name(f)
            data_dict[basename] = f
        except RuntimeError as e:
            print(e)
        except:
            print(f"ERROR: file={f}")

    for f in patch_files:
        info = get_patch_info(f)
        if info is None:
            continue
        basename,bbox,label = info
        if basename not in data_dict:
            print(f"Find: {basename} in data dict faild, patch file {f}")
            continue
        img_path = data_dict[basename]
        xml_path = wmlu.change_suffix(img_path,"xml")
        patch_one_xml_file(bbox,label,xml_path,img_path)








