import wml.img_utils as wmli
import wml.object_detection2.visualization as odv
from wml.iotoolkit.labelmemckeypoints_dataset import LabelmeMCKeypointsDataset
from wml.object_detection2.standard_names import *
import argparse
import os.path as osp
import wml.wml_utils as wmlu

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
    parser.add_argument('--type', type=str, default='LabelmeMCKeypointsDataset',help='Data set type')
    parser.add_argument(
        '--line-width', type=int, default=2, help='line width')
    parser.add_argument(
        '--view-nr', type=int, default=-1, help='view dataset nr.')
    parser.add_argument('--suffix', type=str, help='suffix to output')
    args = parser.parse_args()

    return args

def text_fn(x,scores):
    return x

DATASETS = {}

def register_dataset(type):
    DATASETS[type.__name__] = type

register_dataset(LabelmeMCKeypointsDataset)

def simple_names(x):
    if "--" in x:
        return x.split("--")[-1]
    return x

if __name__ == "__main__":

    args = parse_args()
    view_nr = args.view_nr
    shuffle = view_nr>0
    print(DATASETS,args.type)
    data = DATASETS[args.type](label_text2id=None,shuffle=shuffle)
    data.read_data(args.src_dir,img_suffix=args.ext)

    if view_nr>0:
        data.files = data.files[:view_nr]

    for x in data.get_items():
        kps = x[GT_KEYPOINTS]
        labels = x[GT_LABELS]
        full_path = x[IMG_INFO][FILEPATH]
        img = wmli.decode_img(x[IMAGE])
        print(full_path)
        old_shape = img.shape

        if args.new_width > 1:
            img = wmli.resize_width(img,args.new_width)
            kps = kps.resize(img.shape[:2][::-1])
        elif args.new_height > 1:
            img = wmli.resize_height(img,args.new_height)
            kps = kps.resize(img.shape[:2][::-1])

        img = odv.draw_maskv2(img,classes=labels,masks=kps)
        r_path = wmlu.get_relative_path(full_path,args.src_dir)
        save_path = osp.join(args.out_dir,r_path)
        if args.suffix is not None and len(args.suffix)>0:
            save_path = osp.splitext(save_path)
            save_path = save_path[0]+args.suffix+save_path[1]
        print(save_path)

        wmli.imwrite(save_path,img)
