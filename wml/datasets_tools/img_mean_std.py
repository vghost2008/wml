from argparse import ArgumentParser
import wml.img_utils as wmli
import wml.wml_utils as wmlu
import numpy as np
import os
from wml.iotoolkit.imgs_reader_mt import ImgsReader,MaxImgLongSize
import sys
import random

'''
对图像信息进行统计
'''

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('src_dir', type=str, default="/home/wj/ai/mldata1/B7mura/datas/try_s0",help='source video directory')
    parser.add_argument("--max-long-size",type=int,default=1024,help="max img long size")
    args = parser.parse_args()
    return args

def get_img_mean_std(img):
    img = img.astype(np.float32)
    mean = np.mean(img,axis=(0,1),keepdims=False)
    std = np.std(img,axis=(0,1),keepdims=False)
    return mean,std


def get_imgs_info(files,args):
    max_long_size = args.max_long_size #统计图像像素值信息时，如果图像最长边长于max_long_size，则缩放至max_long_size
    transform = None
    if max_long_size>1:
        transform = MaxImgLongSize(max_long_size)

    for i,file in enumerate(files):
        img = wmli.imread(file)
        if transform is not None:
            img = transform(img)
        sys.stdout.write(f"Process {i}/{len(files)}        \r")
        sys.stdout.flush()
        if len(img)==0:
            print(f"ERROR: Read {file} faild.")
            continue
        try:
            mean,std = get_img_mean_std(img)
            print(mean.shape,std.shape)
            mean = [f"{v:.1f}" for v in mean]
            std = [f"{v:.1f}" for v in std]
            mean = ','.join(mean)
            std = ','.join(std)
            print(f"{file}: mean=[{mean}], std=[{std}]")
        except Exception as e:
            print(f"ERROR: Read {file} faild: {e}")
    
    sys.stdout.flush()

if __name__ == "__main__":
    args = parse_args()
    img_files = wmlu.get_files(args.src_dir,suffix=wmli.BASE_IMG_SUFFIX)
    get_imgs_info(img_files,args)
