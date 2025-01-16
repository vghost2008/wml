from argparse import ArgumentParser
import pickle
import numpy as np
from wml.wfilesystem import recurse_get_subdir_in_dir
import wml.wml_utils as wmlu
import glob
import os.path as osp
import os

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('data_dir', type=str,help='data dir')
    parser.add_argument('--prefix', type=str,default="checkpoint_",help='name')
    parser.add_argument('--suffix', type=str,default=".pth",help='checkpoint suffix')
    args = parser.parse_args()
    return args

def remove_files(f):
    try:
        os.remove(f)
    except Exception as e:
        print(f"ERROR: {e}")

def process_dir(data_dir,prefix,suffix):
    files = glob.glob(osp.join(data_dir,prefix+"*"+suffix))
    if len(files)<=1:
        return 
    prefix_len = len(prefix)
    max_idx = -1
    max_idx_file_path = None
    ckpt_files = []
    for x in files:
        try:
            #name = osp.splitext(osp.basename(x))[0]
            name = osp.basename(x)[:-len(suffix)]
            idx = int(name[prefix_len:])
            if idx>max_idx:
                max_idx = idx
                if max_idx_file_path is not None:
                    ckpt_files.append(max_idx_file_path)
                max_idx_file_path = x
            else:
                ckpt_files.append(x)
        except:
            pass
    print(f"Files to remove")
    wmlu.show_list(ckpt_files)
    for x in ckpt_files:
        remove_files(x)
    print(f"Files to keep")
    print(max_idx_file_path)


if __name__ == "__main__":
    args = parse_args()
    dirs = recurse_get_subdir_in_dir(args.data_dir)
    for dir in dirs:
        dir = osp.join(args.data_dir,dir)
        process_dir(dir,args.prefix,args.suffix)
    
