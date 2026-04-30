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
    parser.add_argument("-ss","--size-step",type=int,default=10,help="size step")
    parser.add_argument("-pv","--pixel-value",action='store_true', help="statistics pixel value")
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

def show_shape_info(shapes,step):

    if len(shapes) == 0:
        return

    widths = wmlu.MDict(dtype=list)
    heights = wmlu.MDict(dtype=list)
    all_widths = []
    all_heights = []
    ratios = []

    for shape in shapes:
        w = shape[1]
        h = shape[0]
        w_key = wmlu.align_to(w,step)
        h_key = wmlu.align_to(h,step)
        all_widths.append(w)
        all_heights.append(h)
        ratios.append(w/max(h,1))
        widths[w_key].append(h)
        heights[h_key].append(w)

    widths = list(widths.items())
    widths.sort(key=lambda x:x[0])
    heights = list(heights.items())
    heights.sort(key=lambda x:x[0])

    print("\n")
    print("="*120)
    print("Width:")
    for k,v in widths:
        print(f"{k:>8}, total {len(v):<5}, {len(v)*100.0/len(shapes):>6.2f}%, height mean: {int(np.mean(v)):<5}, std: {int(np.std(v)):<4}")

    print("\n")
    print("-"*120)
    print("Height:")
    for k,v in heights:
        print(f"{k:>8}, total {len(v):<5}, {len(v)*100.0/len(shapes):>6.2f}%, height mean: {int(np.mean(v)):<5}, std: {int(np.std(v)):<4}")

    print("\n")
    print("-"*120)
    print(f" WIDTH: max={np.max(all_widths):5.0f}, min={np.min(all_widths):5.0f}, mean={np.mean(all_widths):5.0f}, std={np.std(all_widths):5.0f}")
    print(f"HEIGHT: max={np.max(all_heights):5.0f}, min={np.min(all_heights):5.0f}, mean={np.mean(all_heights):5.0f}, std={np.std(all_heights):5.0f}")
    print(f"RATIOS(W/H): max={np.max(ratios):<4.2f}, min={np.min(ratios):<4.2f}, mean={np.mean(ratios):<4.2f}, std={np.std(ratios):4.2f}")
    print("="*120)
    print("\n")

def get_imgs_info(files,args):
    value = []
    contrast = []
    max_long_size = args.max_long_size #统计图像像素值信息时，如果图像最长边长于max_long_size，则缩放至max_long_size
    size_step = args.size_step #对宽高进行计数时通过size_step把不同的值划分到不同的bucket
    transform = None
    if max_long_size>1:
        transform = MaxImgLongSize(max_long_size)
    reader = ImgsReader(files,thread_nr=8,transform=transform)
    shapes = []

    for i,(file,img) in enumerate(reader):
        sys.stdout.write(f"Process {i}/{len(reader)}        \r")
        sys.stdout.flush()
        if len(img)==0:
            print(f"ERROR: Read {file} faild.")
            continue
        try:
            shape = wmli.get_img_size(file)
            shapes.append(shape)
            value.append(np.mean(img,axis=(0,1)))
            contrast.append(get_img_contrast_info(img))
        except Exception as e:
            print(f"ERROR: Read {file} faild: {e}")
    
    if len(shapes)==0:
        return
    contrast = np.array(contrast)

    print("\n")
    show_shape_info(shapes, step=size_step)
    print(f"Pixel Value: max={np.max(value,axis=(0,))}\n min={np.min(value,axis=(0,))}\n mean={np.mean(value,axis=(0,))}\n std={np.std(value,axis=(0,))}")
    print(f"Contrast: max={np.max(contrast,axis=(0,))}\n min={np.min(contrast,axis=(0,))}\n mean={np.mean(contrast,axis=(0,))}\n std={np.std(contrast,axis=(0,))}")
    sys.stdout.flush()


def get_sample_imgs_info(files,args):
    size_step = args.size_step #对宽高进行计数时通过size_step把不同的值划分到不同的bucket
    shapes = []

    for i,file in enumerate(files):
        sys.stdout.write(f"Process {i}/{len(files)}        \r")
        sys.stdout.flush()
        try:
            shape = wmli.get_img_size(file)
            shapes.append(shape)
        except Exception as e:
            print(f"ERROR: Read {file} faild: {e}")
    
    if len(shapes)==0:
        return
    show_shape_info(shapes,step=size_step)
    sys.stdout.flush()

if __name__ == "__main__":
    #os.environ['CUDA_VISIBLE_DEVICES'] = "3"
    #os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    args = parse_args()
    img_files = wmlu.get_files(args.src_dir,suffix=".jpg;;.jpeg;;.png;;.bmp;;.tif;;.mci")
    if args.test_nr is not None and args.test_nr>0:
        print(f"Only test {args.test_nr} imgs.")
        random.shuffle(img_files)
        img_files = img_files[:args.test_nr]
    if not args.pixel_value:
        get_sample_imgs_info(img_files,args)
    else:
        get_imgs_info(img_files,args)
