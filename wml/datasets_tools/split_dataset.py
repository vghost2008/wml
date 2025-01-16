import argparse
import wml.wml_utils as wmlu
import os.path as osp
import os
import math
import random
import time
import copy
import shutil

'''
主要用于拆分训练集，测试集等场景

将数据集按split指定的比例拆分到不同的目录中
也可以使用绝对的数量，使用绝对数量时，-1表示剩余的其它文件

sub-dir为True时表示，百分比对一级子目录当独计算，否则对所有目标统一计算
'''

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
        default=".jpg;;.jpeg;;.bmp;;.png;;.tif",
        help='img suffix')
    parser.add_argument(
        '--split',
        type=float,
        nargs="+",
        default=[0.9,0.1],
        help='split percent')
    parser.add_argument(
        '--max-nr',
        type=int,
        help='split set max nr')
    parser.add_argument(
        '--allow-empty',
        action='store_true',
        help='include img files without annotation')
    parser.add_argument('--sub-dir', action='store_true',help='whether to sample data in sub dirs.')
    args = parser.parse_args()

    return args

def copy_files(files,save_dir,add_nr,src_dir):
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    for i,(imgf,annf) in enumerate(files):
        #basename = wmlu.base_name(imgf)
        basename = wmlu.get_relative_path(imgf,src_dir)
        basename = osp.splitext(basename)[0]
        if add_nr:
            basename = basename+f"_{i}"
        suffix = osp.splitext(imgf)[1]
        save_path = osp.join(save_dir,basename+suffix)
        cur_save_dir = osp.dirname(save_path)
        os.makedirs(cur_save_dir,exist_ok=True)

        print(imgf,"--->",save_path)
        shutil.copy(imgf,save_path)
        suffix = osp.splitext(annf)[1]
        print(annf,"--->",osp.join(save_dir,basename+suffix))
        shutil.copy(annf,osp.join(save_dir,basename+suffix))


def split_one_dir(src_dir,out_dir,splits,args,sub_dir_name=None):

    print(f"Process {src_dir}, save dir {out_dir}")
    splits = copy.deepcopy(splits)
    max_nr = args.max_nr

    if sub_dir_name is not None:
        src_dir = osp.join(src_dir,sub_dir_name)

    img_files = wmlu.get_files(src_dir,suffix=args.img_suffix)
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
    if sub_dir_name is None:
        save_dir = wmlu.get_unused_path(out_dir)
    else:
        save_dir = out_dir
    os.makedirs(save_dir,exist_ok=True)
    random.seed(int(time.time()))
    random.shuffle(all_files)
    print(f"Find {len(all_files)} files in {src_dir}")

    use_percent = True
    total_nr = 0
    for v in splits:
        if v<0 or v>=1:
            use_percent = False
            if v>0:
                total_nr += v
    old_splits = copy.deepcopy(splits)
    for i,v in enumerate(splits):
        if v<0:
            splits[i] = len(all_files)-total_nr
            print(f"Update splits from {old_splits} to {splits}")
            break
    

    for i,v in enumerate(splits):
        if sub_dir_name is not None:
            t_save_dir = osp.join(save_dir,"data_"+str(old_splits[i]),sub_dir_name)
        else:
            t_save_dir = osp.join(save_dir,"data_"+str(v))
        if i<len(splits)-1:
            if use_percent:
                t_nr = int(v*len(all_files)+0.5)
            else:
                t_nr = int(v)
            tmp_files = all_files[:t_nr]
            all_files = all_files[t_nr:]
        else:
            tmp_files = all_files
            t_nr = len(tmp_files)
        print(f"split {v} as {t_nr} files")
        wmlu.show_list(tmp_files)
        if max_nr is not None and max_nr>0:
            tmp_files = list(tmp_files)
            random.shuffle(tmp_files)
            tmp_files = tmp_files[:max_nr]
        copy_files(tmp_files,t_save_dir,add_nr,src_dir=src_dir)


if __name__ == "__main__":
    args = parse_args()
    splits = args.split
    if splits[0]>0 and splits[0]<1:
        sum = 0.0
        for x in splits:
            sum += x
        if math.fabs(sum-1.0)>0.01:
            print(f"Error split, sum(split)==1")
            exit(-1)
    
    if args.sub_dir:
        for sub_dir in wmlu.get_subdir_in_dir(args.src_dir):
            '''cur_src_dir = osp.join(args.src_dir,sub_dir)
            cur_out_dir = osp.join(args.out_dir,sub_dir)
            split_one_dir(cur_src_dir,cur_out_dir,splits,args)'''
            split_one_dir(args.src_dir,args.out_dir,splits,args,sub_dir_name=sub_dir)
    else:
        split_one_dir(args.src_dir,args.out_dir,splits,args)
