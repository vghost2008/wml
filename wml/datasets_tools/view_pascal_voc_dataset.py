import wml.img_utils as wmli
import wml.object_detection2.visualization as odv
import matplotlib.pyplot as plt
from wml.iotoolkit.pascal_voc_toolkit import PascalVOCData
import argparse
import os.path as osp
import wml.wml_utils as wmlu
import wml.wtorch.utils as wtu

def parse_args():
    parser = argparse.ArgumentParser(description='extract optical flows')
    parser.add_argument('src_dir', type=str, help='source video directory')
    parser.add_argument('out_dir', type=str, help='output rawframe directory')
    parser.add_argument(
        '--ext',
        type=str,
        default='jpg',
        #choices=['avi', 'mp4', 'webm','MOV'],
        help='video file extensions')
    parser.add_argument(
        '--new-width', type=int, default=0, help='resize image width')
    parser.add_argument(
        '--new-height', type=int, default=0, help='resize image height')
    args = parser.parse_args()

    return args

def text_fn(x,scores):
    return x

if __name__ == "__main__":

    args = parse_args()
    data = PascalVOCData(label_text2id=None,shuffle=False,absolute_coord=True)
    data.read_data(args.src_dir,img_suffix=args.ext)

    for x in data.get_items():
        full_path, img_info,category_ids, category_names, boxes,*_ =  x
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
            text_fn=text_fn, thickness=1,
            show_text=True,
            font_scale=0.8,
            is_relative_coordinate=False)

        save_path = osp.join(args.out_dir,wmlu.base_name(full_path)+".jpeg")

        wmli.imwrite(save_path,img)
