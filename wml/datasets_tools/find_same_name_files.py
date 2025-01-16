from argparse import ArgumentParser
import wml.img_utils as wmli
import wml.wml_utils as wmlu
import numpy as np
import os
import os.path as osp
from wml.iotoolkit.imgs_reader_mt import ImgsReader,MaxImgLongSize
import sys
import random
import shutil

'''
在同一目录下查找相同文件名的文件并打印其路径
'''

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('src_dir', type=str, default=["/home/wj/ai/mldata1/B7mura/datas/try_s0"],nargs="+",help='source video directory')
    parser.add_argument('--ext', type=str, help='files ext')
    parser.add_argument('--save-dir', type=str, help='save dir')
    args = parser.parse_args()
    return args



def find_same_name_files(files,save_dir=None):
    name2path = wmlu.MDict(dtype=list)
    files_nr = 0
    for file,root_path in files:
        name = os.path.basename(file)
        #rpath = wmlu.get_relative_path(file,root_path)
        name2path[name].append((file,root_path))
        files_nr += 1
    print(f"Total find {files_nr} files")
    same_nr = 0
    different_nr = 0
    
    for k,v in name2path.items():
        if len(v)==1:
            different_nr += 1
            continue
        files_v = [x[0] for x in v]
        print_info(k,files_v)
        if save_dir is not None:
            save_files(v,save_dir)

        same_nr += 1
    
    print(f"Same nr: {same_nr}, different nr {different_nr}")

def print_info(name,files):
    print("")
    print(name)
    sizes = []
    for file in files:
        sizes.append(os.path.getsize(file))
    sizes = set(sizes)
    if len(sizes)<len(files):
        for i,file in enumerate(files):
            print(f" ->{file} {os.path.getsize(file)} {wmlu.file_md5(file)}")
            #print(f"remove {file}")
            #os.remove(file)
    else:
        for i,file in enumerate(files):
            print(f" ->{file} {os.path.getsize(file)}")

def save_files(files,save_dir):
    for file_path,root_path in files:
        rp = wmlu.get_relative_path(file_path,root_path)
        sp = osp.join(save_dir,rp)
        if osp.exists(sp):
            sp = wmlu.get_unused_path_with_suffix(sp)
        dir_name = osp.dirname(sp)
        if not osp.exists(dir_name):
            os.makedirs(dir_name)
        
        shutil.copy(file_path,sp)
        print(f"Copy {file_path} --> {sp}")

if __name__ == "__main__":
    args = parse_args()
    all_files = []
    for sd in args.src_dir:
        root_path = os.path.abspath(sd)
        img_files = wmlu.get_files(root_path,suffix=args.ext)
        cur_files = [(img_f,root_path) for img_f in img_files]
        all_files.extend(cur_files)
    find_same_name_files(all_files,args.save_dir)
