#coding=utf-8
import sys
import os
import wml.object_detection2.npod_toolkit as npod
import numpy as np
import wml.object_detection2.visualization as odv
import wml.img_utils as wmli
from wml.iotoolkit.pascal_voc_toolkit import PascalVOCData,read_voc_xml,write_voc_xml
from wml.iotoolkit.coco_toolkit import COCOData
from wml.iotoolkit.labelme_toolkit import LabelMeData
import wml.object_detection2.bboxes as odb 
import wml.wml_utils as wmlu
from wml.iotoolkit.mapillary_vistas_toolkit import MapillaryVistasData
from functools import partial
from argparse import ArgumentParser
from itertools import count
import os.path as osp
import wml.img_utils as wmli
import shutil

'''
基于文件的标注xml补丁脚本
补丁文件名的命名规则为：
    [org_file_name]_bbox{xmin},{ymin},{xmax},{ymax}_{label}.jpg
    Example:
    765K230022C1A_T380A0N_SCANIMAGE_20220408_174713_494_bbox1737,551,1753,562_MP1U.jpg
打补丁时通过文件名org_file_name找到原文件，并把补丁文件名中包含的标注信息插入到原标注文件中
'''
def parse_args():
    parser = ArgumentParser()
    parser.add_argument('patch_dir', type=str, help='patch annotation directory')
    parser.add_argument('data_dir', type=str, help='data dir')
    parser.add_argument('--type', type=str, default="xml",help='dataset type')
    parser.add_argument('--save-dir', type=str, help='dir to save annotation views.')
    parser.add_argument('--max-size', type=int, help='max size to save img.')
    args = parser.parse_args()
    return args

def patch_one_xml_file(bbox,label,xml_path,img_path,save_dir=None,max_size=None):
    print(f"Patch {xml_path}")
    shape, bboxes, labels_text, difficult, truncated, probs = read_voc_xml(xml_path,absolute_coord=True)
    bk_xml_path = wmlu.change_suffix(xml_path,"bk")
    shutil.move(xml_path,bk_xml_path)
    if save_dir is not None:
        basename = wmlu.base_name(xml_path)
        os.makedirs(save_dir,exist_ok=True)
        save_img_path = osp.join(save_dir,basename+"_old.jpg")
        img = wmli.imread(img_path)
        if not osp.exists(save_img_path):
            oimg = odv.draw_bboxes(img.copy(),
                                   labels_text,bboxes=bboxes,show_text=True,
                                   is_relative_coordinate=False)
            if max_size is not None:
                oimg = wmli.resize_long_size(oimg,max_size)
            wmli.imwrite(save_img_path,oimg)

    bbox = odb.npchangexyorder([bbox])
    bboxes = np.concatenate([bboxes,bbox],axis=0)
    labels_text.append(label)
    if save_dir is not None:
        save_img_path = osp.join(save_dir,basename+"_new.jpg")
        save_img_path = wmlu.get_unused_path_with_suffix(save_img_path)
        if not osp.exists(save_img_path):
            nimg = odv.draw_bboxes(img.copy(),
                                   labels_text,bboxes=bboxes,show_text=True,
                                   is_relative_coordinate=False)
            if max_size is not None:
                nimg = wmli.resize_long_size(nimg,max_size)
            wmli.imwrite(save_img_path,nimg)
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
    save_dir = args.save_dir #保存patch前及后的对比效果图
    max_size = args.max_size
    patch_files = wmlu.get_files(patch_dir,suffix=wmli.BASE_IMG_SUFFIX)
    data_files = wmlu.get_files(data_dir,suffix=wmli.BASE_IMG_SUFFIX)
    data_dict = wmlu.EDict()
    error_dict = set()

    if save_dir is not None:
        wmlu.create_empty_dir_remove_if(save_dir)

    for f in data_files:
        try:
            basename = wmlu.base_name(f)
            data_dict[basename] = f
        except RuntimeError as e:
            print(e)
        except:
            error_dict.add(basename)
            print(f"ERROR: file={f}")

    for i,f in enumerate(patch_files):
        info = get_patch_info(f)
        if info is None:
            continue
        basename,bbox,label = info
        if basename in error_dict:
            print(f"{basename} is error file, skip.")
            continue
        if basename not in data_dict:
            print(f"Find: {basename} in data dict faild, patch file {f}")
            continue
        img_path = data_dict[basename]
        xml_path = wmlu.change_suffix(img_path,"xml")
        sys.stdout.write(f"{i}/{len(patch_files)} ")
        patch_one_xml_file(bbox,label,xml_path,img_path,save_dir,max_size=max_size)








