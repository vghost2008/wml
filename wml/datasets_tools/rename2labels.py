import argparse
import wml.wml_utils as wmlu
import os.path as osp
import os
import math
import random
import time
from wml.iotoolkit.pascal_voc_toolkit import read_voc_xml
from wml.iotoolkit.labelme_toolkit import read_labelme_data
from wml.iotoolkit import get_auto_dataset_suffix
import shutil

'''
把数据集中的文件重命名为相应的标签
'''

def parse_args():
    parser = argparse.ArgumentParser(description='split dataset')
    parser.add_argument('src_dir', type=str, help='source video directory')
    parser.add_argument('out_dir', type=str, help='output rawframe directory')
    parser.add_argument(
        '--suffix',
        type=str,
        default='auto',
        choices=['json', 'xml', 'txt','auto'],
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

def copy_files(imgf,annf,save_dir,labels,silent=False):

    suffix = osp.splitext(imgf)[1]
    save_path = osp.join(save_dir,labels+suffix)
    save_path = wmlu.get_unused_path_with_suffix(save_path)
    if osp.exists(imgf):
        if not silent:
            print(imgf,"--->",save_path)
        shutil.copy(imgf,save_path)
    suffix = osp.splitext(annf)[1]
    if not silent:
        save_path = wmlu.change_suffix(save_path,suffix[1:])
        print(annf,"--->",save_path)
    shutil.copy(annf,save_path)

def get_labels(ann_file,suffix):
    if suffix == "xml":
        labels = read_voc_xml(ann_file)[2]
    elif suffix == "json":
        image,annotation_list = read_labelme_data(ann_file,label_text_to_id=None,mask_on=False)
        labels = [x['category_id'] for x in annotation_list]
    
    if len(labels)==0:
        labels = ['NONE']

    return labels



if __name__ == "__main__":
    args = parse_args()
    if args.suffix == "auto":
        args.suffix = get_auto_dataset_suffix(args.src_dir)

    if not args.no_imgs:
        img_files = wmlu.get_files(args.src_dir,suffix=args.img_suffix)
        ann_files = [wmlu.change_suffix(x,args.suffix) for x in img_files]
    else:
        ann_files = wmlu.get_files(args.src_dir,suffix=args.suffix)
        img_files = [wmlu.change_suffix(x,"jpg") for x in ann_files]
    basenames = [wmlu.base_name(x) for x in img_files]
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
        labels = list(set(get_labels(ann_f,args.suffix)))
        labels.sort()
        if len(labels)==0:
            labels = "NONE"
        elif len(labels)==1:
            labels = labels[0]
        else:
            labels = "_".join(labels)
        copy_files(img_f,ann_f,save_dir,labels,silent=args.silent)


