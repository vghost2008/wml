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
    parser.add_argument("--test-nr",type=int,help="max imgs to test")
    parser.add_argument("--max-long-size",type=int,default=1024,help="max img long size")
    parser.add_argument("--size-step",type=int,default=10,help="size step")
    args = parser.parse_args()
    return args

def get_img_contrast_info(img):
    img = img.astype(np.float32)
    mean = np.mean(img,axis=(0,1),keepdims=True)
    contrast = np.abs((img-mean)/(mean+1e-2))
    min_contrast = np.min(contrast,axis=(0,1))
    max_contrast = np.max(contrast,axis=(0,1))
    contrast = np.mean(contrast,axis=(0,1))
    contrast = np.concatenate([contrast,min_contrast,max_contrast],axis=0)
    return contrast


def get_imgs_info(files,args):
    widths = []
    heights = []
    value = []
    contrast = []
    max_long_size = args.max_long_size #统计图像像素值信息时，如果图像最长边长于max_long_size，则缩放至max_long_size
    size_step = args.size_step #对宽高进行计数时通过size_step把不同的值划分到不同的bucket
    transform = None
    if max_long_size>1:
        transform = MaxImgLongSize(max_long_size)
    reader = ImgsReader(files,thread_nr=8,transform=transform)
    width_counter = wmlu.Counter()
    height_counter = wmlu.Counter()

    for i,(file,img) in enumerate(reader):
        sys.stdout.write(f"Process {i}/{len(reader)}        \r")
        sys.stdout.flush()
        if len(img)==0:
            print(f"ERROR: Read {file} faild.")
            continue
        try:
            shape = wmli.get_img_size(file)
            widths.append(shape[1])
            heights.append(shape[0])
            width_counter.add(int(shape[1]//size_step)*size_step)
            height_counter.add(int(shape[0]//size_step)*size_step)
            value.append(np.mean(img,axis=(0,1)))
            contrast.append(get_img_contrast_info(img))
        except Exception as e:
            print(f"ERROR: Read {file} faild: {e}")
    
    if len(widths)==0:
        return
    widths = np.array(widths)
    heights = np.array(heights)
    value = np.array(value)
    contrast = np.array(contrast)
    ratios = widths/(heights+1e-5)

    print("\n")
    print("width_counter")
    width_counter = list(width_counter.items())
    width_counter = sorted(width_counter,key=lambda x:x[0])
    wmlu.show_list(width_counter)
    print("height_counter")
    height_counter = list(height_counter.items())
    height_counter = sorted(height_counter,key=lambda x:x[0])
    wmlu.show_list(height_counter)
    print(f"WIDTH: max={np.max(widths)}, min={np.min(widths)}, mean={np.mean(widths)}, std={np.std(widths)}")
    print(f"HEIGHT: max={np.max(heights)}, min={np.min(heights)}, mean={np.mean(heights)}, std={np.std(heights)}")
    print(f"RATIOS(W/H): max={np.max(ratios)}, min={np.min(ratios)}, mean={np.mean(ratios)}, std={np.std(ratios)}")
    print(f"Pixel Value: max={np.max(value,axis=(0,))}\n min={np.min(value,axis=(0,))}\n mean={np.mean(value,axis=(0,))}\n std={np.std(value,axis=(0,))}")
    print(f"Contrast: max={np.max(contrast,axis=(0,))}\n min={np.min(contrast,axis=(0,))}\n mean={np.mean(contrast,axis=(0,))}\n std={np.std(contrast,axis=(0,))}")
    sys.stdout.flush()

if __name__ == "__main__":
    #os.environ['CUDA_VISIBLE_DEVICES'] = "3"
    #os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    args = parse_args()
    img_files = wmlu.get_files(args.src_dir,suffix=".jpg;;.jpeg;;.png;;.bmp;;.tif")
    if args.test_nr is not None and args.test_nr>0:
        print(f"Only test {args.test_nr} imgs.")
        random.shuffle(img_files)
        img_files = img_files[:args.test_nr]
    get_imgs_info(img_files,args)
