import argparse
import wml_utils as wmlu
import os.path as osp
import os
import math
import random
import time

def parse_args():
    parser = argparse.ArgumentParser(description='split dataset')
    parser.add_argument('src_dir', type=str, help='source video directory')
    parser.add_argument('out_dir', type=str, help='output rawframe directory')
    parser.add_argument(
        '--suffix',
        type=str,
        default='json',
        choices=['json', 'xml', 'txt'],
        help='annotation suffix')
    parser.add_argument(
        '--img-suffix',
        type=str,
        default=".jpg;;.jpeg;;.bmp;;.png",
        help='img suffix')
    parser.add_argument(
        '--split',
        type=float,
        nargs="+",
        default=[0.9,0.1],
        help='split percent')
    parser.add_argument(
        '--allow-empty',
        action='store_true',
        help='include img files without annotation')
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
        suffix = osp.splitext(annf)[1]
        print(annf,"--->",osp.join(save_dir,basename+suffix))
        wmlu.try_link(annf,osp.join(save_dir,basename+suffix))

if __name__ == "__main__":
    args = parse_args()
    splits = args.split
    sum = 0.0
    for x in splits:
        sum += x
    if math.fabs(sum-1.0)>0.01:
        print(f"Error split, sum(split)==1")
        exit(-1)
    img_files = wmlu.get_files(args.src_dir,suffix=args.img_suffix)
    ann_files = [wmlu.change_suffix(x,args.suffix) for x in img_files]
    basenames = [wmlu.base_name(x) for x in img_files]
    if len(basenames) == len(set(basenames)):
        add_nr = False
    else:
        add_nr = True
    all_files = list(zip(img_files,ann_files))
    if not args.allow_empty:
        all_files = list(filter(lambda x:osp.exists(x[1]),all_files))
    save_dir = wmlu.get_unused_path(args.out_dir)
    os.makedirs(save_dir)
    random.seed(int(time.time()))
    random.shuffle(all_files)
    print(f"Find {len(all_files)} files")

    for i,v in enumerate(splits):
        t_save_dir = osp.join(save_dir,"data_"+str(v))
        if i<len(splits)-1:
            t_nr = int(v*len(all_files)+0.5)
            tmp_files = all_files[:t_nr]
            all_files = all_files[t_nr:]
        else:
            tmp_files = all_files
            t_nr = len(tmp_files)
        print(f"split {v} {t_nr} files")
        wmlu.show_list(tmp_files)
        copy_files(tmp_files,t_save_dir,add_nr)

