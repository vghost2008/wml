import argparse
import wml_utils as wmlu
import os.path as osp
import os
import math
import random
import time
from iotoolkit.pascal_voc_toolkit import read_voc_xml
from iotoolkit.labelme_toolkit import read_labelme_data

def parse_args():
    parser = argparse.ArgumentParser(description='split dataset')
    parser.add_argument('src_dir', type=str, help='source video directory')
    parser.add_argument('out_dir', type=str, help='output rawframe directory')
    parser.add_argument(
        '--suffix',
        type=str,
        default='xml',
        choices=['json', 'xml', 'txt'],
        help='annotation suffix')
    parser.add_argument(
        '--img-suffix',
        type=str,
        default=".jpg;;.jpeg;;.bmp;;.png",
        help='img suffix')
    parser.add_argument(
        '--allow-empty',
        action='store_true',
        help='include img files without annotation')
    parser.add_argument(
        '--no-imgs',
        action='store_true',
        help='only split xmls')
    parser.add_argument(
        '--silent',
        action='store_true',
        help='silent')
    args = parser.parse_args()

    args = parser.parse_args()

    return args

def copy_files(imgf,annf,save_dir,add_nr,silent=False):
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    basename = wmlu.base_name(imgf)
    if add_nr:
        basename = basename+f"_{i}"
    suffix = osp.splitext(imgf)[1]
    if osp.exists(imgf):
        if not silent:
            print(imgf,"--->",osp.join(save_dir,basename+suffix))
        wmlu.try_link(imgf,osp.join(save_dir,basename+suffix))
    suffix = osp.splitext(annf)[1]
    if not silent:
        print(annf,"--->",osp.join(save_dir,basename+suffix))
    wmlu.try_link(annf,osp.join(save_dir,basename+suffix))

def get_labels(ann_file,suffix):
    if suffix == "xml":
        return read_voc_xml(ann_file)[2]
    elif suffix == "json":
        image,annotation_list = read_labelme_data(ann_file,label_text_to_id=None,mask_on=False)
        labels = [x['category_id'] for x in annotation_list]
        return labels



if __name__ == "__main__":
    args = parse_args()
    if not args.no_imgs:
        img_files = wmlu.get_files(args.src_dir,suffix=args.img_suffix)
        ann_files = [wmlu.change_suffix(x,args.suffix) for x in img_files]
    else:
        ann_files = wmlu.get_files(args.src_dir,suffix=args.suffix)
        img_files = [wmlu.change_suffix(x,"jpg") for x in ann_files]
    basenames = [wmlu.base_name(x) for x in img_files]
    if len(basenames) == len(set(basenames)):
        add_nr = False
    else:
        add_nr = True
    all_files = list(zip(img_files,ann_files))
    wmlu.show_list(all_files[:100])
    if len(all_files)>100:
        print("...")
    if not args.allow_empty:
        all_files = list(filter(lambda x:osp.exists(x[1]),all_files))
    save_dir = wmlu.get_unused_path(args.out_dir)
    os.makedirs(save_dir)
    random.seed(int(time.time()))
    random.shuffle(all_files)
    print(f"Find {len(all_files)} files")

    for i,(img_f,ann_f) in enumerate(all_files):
        labels = get_labels(ann_f,args.suffix)
        for l in set(labels):
            t_save_dir = osp.join(save_dir,l)
            os.makedirs(t_save_dir,exist_ok=True)
            copy_files(img_f,ann_f,t_save_dir,add_nr,silent=args.silent)


