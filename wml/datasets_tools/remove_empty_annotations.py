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
from wml.iotoolkit.image_folder import ImageFolder
import wml.img_utils as wmli
import shutil

'''
将一个数据集中除指定的labels外的数据拷贝到输出目录
'''

def parse_args():
    parser = argparse.ArgumentParser(description='split dataset')
    parser.add_argument('src_dir', type=str, help='source video directory')
    parser.add_argument(
        '--suffix',
        type=str,
        default='auto',
        choices=['json', 'xml', 'txt','auto'],
        help='annotation suffix')
    parser.add_argument(
        '--no-imgs',
        action='store_true',
        help='only split xmls')
    args = parser.parse_args()

    return args

def get_labels(ann_file,suffix):
    if suffix == "xml":
        labels = read_voc_xml(ann_file)[2]
    elif suffix == "json":
        image,annotation_list = read_labelme_data(ann_file,label_text_to_id=None,mask_on=False)
        labels = [x['category_id'] for x in annotation_list]
    elif suffix == "none":
        labels = [ImageFolder.get_label(ann_file)]
    if len(labels)==0:
        labels = ['NONE']

    return labels



if __name__ == "__main__":
    args = parse_args()
    if args.suffix == "auto":
        args.suffix = get_auto_dataset_suffix(args.src_dir)
    
    if args.suffix == "none":
        args.no_imgs = False
        args.allow_empty = True

    if not args.no_imgs:
        img_files = wmlu.get_files(args.src_dir,suffix=wmli.BASE_IMG_SUFFIX)
        ann_files = [wmlu.change_suffix(x,args.suffix) for x in img_files]
    else:
        ann_files = wmlu.get_files(args.src_dir,suffix=args.suffix)
        img_files = [wmlu.change_suffix(x,"jpg") for x in ann_files]
    all_files = list(zip(img_files,ann_files))
    wmlu.show_list(all_files[:100])
    if len(all_files)>100:
        print("...")
    all_files = list(filter(lambda x:osp.exists(x[1]),all_files))

    random.seed(int(time.time()))
    random.shuffle(all_files)
    print(f"Find {len(all_files)} files")

    rm_labels = set(args.labels)

    total_skip = 0
    total_copy = 0

    files2remove = []
    for i,(img_f,ann_f) in enumerate(all_files):
        labels = get_labels(ann_f,args.suffix)
        if len(labels) == 0:
            files2remove.append(ann_f)
    
    if len(files2remove) == 0:
        print(f"No files to remove")
        exit(0)
    
    print(f"Files to remove")
    wmlu.show_list(files2remove)
    ans = input(f"Remove {len(files2remove)} files [Y/N]?\n")
    if ans.lower() == "y":
        for f in files2remove:
            print(f"Remove {f}")
            os.remove(f)