import img_utils as wmli
import object_detection2.visualization as odv
import matplotlib.pyplot as plt
from iotoolkit.pascal_voc_toolkit import PascalVOCData
from iotoolkit.mapillary_vistas_toolkit import MapillaryVistasData
from iotoolkit.coco_toolkit import COCOData
from iotoolkit.labelme_toolkit import LabelMeData
import argparse
import os.path as osp
import wml_utils as wmlu
import wtorch.utils as wtu

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
    parser.add_argument('--type', type=str, default='PascalVOCData',help='Data set type')
    parser.add_argument(
        '--line-width', type=int, default=2, help='line width')
    parser.add_argument(
        '--view-nr', type=int, default=-1, help='view dataset nr.')
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

def simple_names(x):
    if "--" in x:
        return x.split("--")[-1]
    return x

if __name__ == "__main__":

    args = parse_args()
    view_nr = args.view_nr
    shuffle = view_nr>0
    print(DATASETS,args.type)
    data = DATASETS[args.type](label_text2id=None,shuffle=shuffle,absolute_coord=True)
    data.read_data(args.src_dir,img_suffix=args.ext)

    if view_nr>0:
        data.files = data.files[:view_nr]

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


        img = odv.draw_bboxes(
            img=img, classes=category_names, scores=None, 
            bboxes=boxes, 
            color_fn=None,
            text_fn=text_fn, thickness=args.line_width,
            show_text=True,
            font_scale=0.8,
            is_relative_coordinate=False,
            is_crowd=is_crowd)

        if binary_masks is not None:
            img = odv.draw_maskv2(img,category_names,boxes,binary_masks,is_relative_coordinate=False)

        save_path = osp.join(args.out_dir,wmlu.base_name(full_path)+".jpeg")
        print(save_path)

        wmli.imwrite(save_path,img)
