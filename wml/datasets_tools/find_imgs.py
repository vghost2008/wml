from argparse import ArgumentParser
import wml.img_utils as wmli
import wml.wml_utils as wmlu
import numpy as np
import os
from wml.iotoolkit.imgs_reader_mt import ImgsReader,MaxImgLongSize
import sys
import random
import shutil

'''
筛选符合条件的图像，并打印其路径
条件按需要进行硬编码
'''

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('src_dir', type=str, default="/home/wj/ai/mldata1/B7mura/datas/try_s0",help='source video directory')
    parser.add_argument('save_dir', type=str, help='dir to save data')
    parser.add_argument(
        '--move',
        action='store_true',
        help='Move files.')
    parser.add_argument("--test-nr",type=int,help="max imgs to test")
    args = parser.parse_args()
    return args

def in_range(v,min_v,max_v):
    if v>=min_v and v<=max_v:
        return True
    return False

def apply_file(file,save_path,args):
    if not args.move:
        shutil.copy(file,save_path)
        print(f"copy {file} --> {save_path}")
    else:
        shutil.move(file,save_path)
        print(f"move {file} --> {save_path}")

def get_imgs_info(files,args):
    reader = ImgsReader(files,thread_nr=8,transform=None)
    save_dir = args.save_dir
    if save_dir is not None:
        os.makedirs(save_dir,exist_ok=True)
    ann_suffix = ["json","xml"]

    for i,(file,img) in enumerate(reader):
        sys.stdout.write(f"Process {i}/{len(reader)}        \r")
        sys.stdout.flush()
        if len(img)==0:
            print(f"ERROR: Read {file} faild.")
            continue
        try:
            shape = wmli.get_img_size(file)
            #条件判断
            width = shape[1]
            height = shape[0]
            #if in_range(width,0,224) and in_range(height,0,224):
                #continue
            if width<=600 or height<=600:
                continue
            print(f"{file} shape={shape}")
            if save_dir is not None:
                name = wmlu.get_relative_path(file,args.src_dir)
                save_path = os.path.join(save_dir,name)
                dir = os.path.dirname(save_path)
                os.makedirs(dir,exist_ok=True)
                apply_file(file,save_path,args)
                for ann_s in ann_suffix:
                    cur_file = wmlu.change_suffix(file,ann_s)
                    cur_save_path = wmlu.change_suffix(save_path,ann_s)
                    if os.path.exists(cur_file):
                        apply_file(cur_file,cur_save_path,args)
        except Exception as e:
            print(f"ERROR: Read {file} faild: {e}")
    
    sys.stdout.flush()

if __name__ == "__main__":
    args = parse_args()
    img_files = wmlu.get_files(args.src_dir,suffix=wmli.BASE_IMG_SUFFIX)
    if args.test_nr is not None and args.test_nr>0:
        print(f"Only test {args.test_nr} imgs.")
        random.shuffle(img_files)
        img_files = img_files[:args.test_nr]
    get_imgs_info(img_files,args)
