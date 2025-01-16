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
    parser.add_argument("--test-nr",type=int,help="max imgs to test")
    parser.add_argument("--max-long-size",type=int,default=1024,help="max img long size")
    parser.add_argument('--save-dir', type=str, help='dir to save data')
    args = parser.parse_args()
    return args

def in_range(v,min_v,max_v):
    if v>=min_v and v<=max_v:
        return True
    return False


def get_imgs_info(files,args):
    max_long_size = args.max_long_size #统计图像像素值信息时，如果图像最长边长于max_long_size，则缩放至max_long_size
    transform = None
    if max_long_size>1:
        transform = MaxImgLongSize(max_long_size)
    reader = ImgsReader(files,thread_nr=8,transform=transform)
    save_dir = args.save_dir
    if save_dir is not None:
        os.makedirs(save_dir,exist_ok=True)

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
            if in_range(width,120,130) and in_range(height,120,130):
                continue
            print(f"{file} shape={shape}")
            if save_dir is not None:
                name = wmlu.get_relative_path(file,args.src_dir)
                save_path = os.path.join(save_dir,name)
                dir = os.path.dirname(save_path)
                os.makedirs(dir,exist_ok=True)
                shutil.copy(file,save_path)
                print(f"{file} --> {save_path}")
        except Exception as e:
            print(f"ERROR: Read {file} faild: {e}")
    
    sys.stdout.flush()

if __name__ == "__main__":
    #os.environ['CUDA_VISIBLE_DEVICES'] = "3"
    #os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    args = parse_args()
    img_files = wmlu.get_files(args.src_dir,suffix=".jpg;;.jpeg;;.png;;.bmp")
    if args.test_nr is not None and args.test_nr>0:
        print(f"Only test {args.test_nr} imgs.")
        random.shuffle(img_files)
        img_files = img_files[:args.test_nr]
    get_imgs_info(img_files,args)
