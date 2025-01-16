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
    parser.add_argument('out_dir', type=str, help='output rawframe directory')
    parser.add_argument(
        '--ext',
        type=str,
        default='.jpg;;.bmp;;.jpeg;;.png',
        #choices=['avi', 'mp4', 'webm','MOV'],
        help='video file extensions')
    parser.add_argument(
        '--new-width', type=int, default=0, help='resize image width')
    parser.add_argument(
        '--new-height', type=int, default=0, help='resize image height')
    parser.add_argument('--type', type=str, default='auto',help='Data set type')
    parser.add_argument(
        '--line-width', type=int, default=2, help='line width')
    parser.add_argument(
        '--view-nr', type=int, default=-1, help='view dataset nr.')
    parser.add_argument('--suffix', type=str, default="_view",help='suffix to output')
    parser.add_argument(
        '--copy-imgs',
        action='store_true',
        help='whether copy raw img to target')
    args = parser.parse_args()

    return args

def text_fn(x,scores):
    return x

DATASETS = {}

def register_dataset(type):
    DATASETS[type.__name__] = type

register_dataset(PascalVOCData)
register_dataset(COCOData)
register_dataset(MapillaryVistasData)
register_dataset(LabelMeData)
register_dataset(FastLabelMeData)

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
    data = dataset_type(label_text2id=None,shuffle=shuffle,absolute_coord=True)
    data.read_data(args.src_dir,img_suffix=args.ext)

    if view_nr>0:
        data.files = data.files[:view_nr]

    if args.copy_imgs:
        if args.suffix is None or len(args.suffix)==0:
            args.suffix = "_view"

    for x in data.get_items():
        full_path, img_info,category_ids, category_names, boxes,binary_masks,area,is_crowd,*_ =  x
        print(full_path)
        category_names = [simple_names(x) for x in category_names]
        img = wmli.imread(full_path)
        old_shape = img.shape

        if args.new_width > 1:
            img = wmli.resize_width(img,args.new_width)
            r = img.shape[0]/old_shape[0]
            boxes = boxes*r
        elif args.new_height > 1:
            img = wmli.resize_height(img,args.new_height)
            r = img.shape[0]/old_shape[0]
            boxes = boxes*r
        else:
            r = None
        
        if r is not None and args.copy_imgs:
            raw_img = img.copy()

        img = odv.draw_bboxes(
            img=img, classes=category_names, scores=None, 
            bboxes=boxes, 
            color_fn=None,
            text_fn=text_fn, thickness=args.line_width,
            show_text=True,
            font_scale=0.8,
            is_relative_coordinate=False,
            is_crowd=is_crowd)

        filename = wmlu.get_relative_path(full_path,args.src_dir)

        if binary_masks is not None:
            if r is not None:
                binary_masks = binary_masks.resize(img.shape[:2][::-1])
            img = odv.draw_maskv2(img,category_names,boxes,binary_masks,is_relative_coordinate=False)
        if args.suffix is not None and len(args.suffix)>0:
            r_filename = osp.splitext(filename)[0]
            save_path = osp.join(args.out_dir,r_filename+args.suffix+osp.splitext(full_path)[-1])
            if args.copy_imgs:
                raw_save_path = osp.join(args.out_dir,filename)
                t_dir_path = osp.dirname(raw_save_path)
                if not osp.exists(t_dir_path):
                    os.makedirs(t_dir_path)
                if r is None:
                    shutil.copy(full_path,raw_save_path)
                else:
                    wmli.imwrite(raw_save_path,raw_img)
        else:
            save_path = osp.join(args.out_dir,filename)
        print(save_path)

        wmli.imwrite(save_path,img)

    print(f"Save dir: {args.out_dir}")
