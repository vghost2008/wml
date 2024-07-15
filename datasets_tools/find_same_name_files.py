from argparse import ArgumentParser
import img_utils as wmli
import wml_utils as wmlu
import numpy as np
import os
from iotoolkit.imgs_reader_mt import ImgsReader,MaxImgLongSize
import sys
import random
import shutil

'''
在同一目录下查找相同文件名的文件并打印其路径
'''

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('src_dir', type=str, default="/home/wj/ai/mldata1/B7mura/datas/try_s0",help='source video directory')
    parser.add_argument('--ext', type=str, help='files ext')
    args = parser.parse_args()
    return args



def find_same_name_files(files,root_path):
    name2path = wmlu.MDict(dtype=list)
    files_nr = 0
    for file in files:
        name = os.path.basename(file)
        #rpath = wmlu.get_relative_path(file,root_path)
        name2path[name].append(file)
        files_nr += 1
    print(f"Total find {files_nr} files")
    same_nr = 0
    different_nr = 0
    
    for k,v in name2path.items():
        if len(v)==1:
            different_nr += 1
            continue
        print_info(k,v,root_path)
        same_nr += 1
    
    print(f"Same nr: {same_nr}, different nr {different_nr}")

def print_info(name,files,root_path):
    print("")
    print(name)
    sizes = []
    for file in files:
        sizes.append(os.path.getsize(file))
    sizes = set(sizes)
    if len(sizes)<len(files):
        for i,file in enumerate(files):
            print(f" ->{file} {os.path.getsize(file)} {wmlu.file_md5(file)}")
    else:
        for i,file in enumerate(files):
            print(f" ->{file} {os.path.getsize(file)}")



if __name__ == "__main__":
    args = parse_args()
    root_path = os.path.abspath(args.src_dir)
    img_files = wmlu.get_files(root_path,suffix=args.ext)
    find_same_name_files(img_files,root_path)
