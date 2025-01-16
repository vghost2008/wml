import argparse
import wml.wml_utils as wmlu
import os.path as osp
import os
import shutil
import random
import time
from wml.iotoolkit.common import get_auto_dataset_suffix

'''
将文件拷贝到多个子目录，每个子目录nr个文件
可以根据需要拷贝相应的标注文件(suffix指定)
'''

def parse_args():
    parser = argparse.ArgumentParser(description='split dataset')
    parser.add_argument('src_dir', type=str, help='source video directory')
    parser.add_argument('out_dir', type=str, help='output rawframe directory')
    parser.add_argument(
        '--suffix',
        type=str,
        default='auto',
        choices=['json', 'xml', 'txt'],
        help='annotation suffix')
    parser.add_argument(
        '--img-suffix',
        type=str,
        default=".jpg;;.jpeg;;.bmp;;.png",
        help='img suffix')
    parser.add_argument(
        '--nr',
        type=int,
        default=10000,
        help='files number per dir')
    parser.add_argument(
        '--dir-nr',
        type=int,
        default=-1,
        help='dir number')
    parser.add_argument(
        '--allow-empty',
        action='store_true',
        help='include img files without annotation')
    parser.add_argument(
        '--shuffle',
        action='store_true',
        help='shuffle imgage files')
    parser.add_argument(
        '--name',
        type=str,
        default="data_",
        help='sub dir name')
    args = parser.parse_args()

    return args

def copy_files(files,save_dir,add_nr):
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    for i,(imgf,annf) in enumerate(files):
        basename = wmlu.base_name(imgf)
        if add_nr:
            basename = basename+f"_{i}"
        suffix = osp.splitext(imgf)[1]
        print(imgf,"--->",osp.join(save_dir,basename+suffix))
        wmlu.try_link(imgf,osp.join(save_dir,basename+suffix))

        if not osp.exists(annf):
            continue
        suffix = osp.splitext(annf)[1]
        print(annf,"--->",osp.join(save_dir,basename+suffix))
        shutil.copy(annf,osp.join(save_dir,basename+suffix))




if __name__ == "__main__":
    args = parse_args()
    nr_per_dir = args.nr
    img_files = wmlu.get_files(args.src_dir,suffix=args.img_suffix)

    if args.suffix == "auto":
        args.suffix = get_auto_dataset_suffix(args.src_dir)

    random.seed(int(time.time()))
    if args.shuffle:
        random.shuffle(img_files)
    else:
        img_files.sort()

    ann_files = [wmlu.change_suffix(x,args.suffix) for x in img_files]
    basenames = [wmlu.base_name(x) for x in img_files]
    if len(basenames) == len(set(basenames)):
        add_nr = False
    else:
        add_nr = True
        print(f"Need to add nr name")
    all_files = list(zip(img_files,ann_files))
    if not args.allow_empty:
        all_files = list(filter(lambda x:osp.exists(x[1]),all_files))
    save_dir = wmlu.get_unused_path(args.out_dir)
    os.makedirs(save_dir)
    print(f"Find {len(all_files)} files")

    if args.dir_nr <= 1:
        all_files = wmlu.list_to_2dlist(all_files,args.nr)
    else:
        all_files = wmlu.list_to_2dlistv2(all_files,args.dir_nr)

    for i,v in enumerate(all_files):
        t_save_dir = osp.join(save_dir,args.name+str(i))
        copy_files(v,t_save_dir,add_nr)

