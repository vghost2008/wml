from argparse import ArgumentParser
import img_utils as wmli
import wml_utils as wmlu
import numpy as np
from iotoolkit.imgs_reader_mt import ImgsReader
import sys

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('src_dir', type=str, default="/home/wj/ai/mldata1/B7mura/datas/try_s0",help='source video directory')
    args = parser.parse_args()
    return args

def get_imgs_info(files):
    widths = []
    heights = []
    value = []
    reader = ImgsReader(files)

    for i,(file,img) in enumerate(reader):
        sys.stdout.write(f"Process {i}/{len(reader)}        ")
        if len(img)==0:
            print(f"ERROR: Read {file} faild.")
            continue
        try:
            widths.append(img.shape[1])
            heights.append(img.shape[0])
            value.append(np.mean(img,axis=(0,1)))
        except Exception as e:
            print(f"ERROR: Read {file} faild: {e}")
    
    if len(widths)==0:
        return
    widths = np.array(widths)
    heights = np.array(heights)
    value = np.array(value)
    ratios = widths/(heights+1e-5)

    print("\n")
    print(f"WIDTH: max={np.max(widths)}, min={np.min(widths)}, mean={np.mean(widths)}, std={np.std(widths)}")
    print(f"HEIGHT: max={np.max(heights)}, min={np.min(heights)}, mean={np.mean(heights)}, std={np.std(heights)}")
    print(f"RATIOS(W/H): max={np.max(ratios)}, min={np.min(ratios)}, mean={np.mean(ratios)}, std={np.std(ratios)}")
    print(f"Pixel Value: max={np.max(value,axis=(0,))}\n min={np.min(value,axis=(0,))}\n mean={np.mean(value,axis=(0,))}\n std={np.std(value,axis=(0,))}")

if __name__ == "__main__":
    args = parse_args()
    img_files = wmlu.get_files(args.src_dir,suffix=".jpg;;.jpeg;;.png;;.bmp")
    get_imgs_info(img_files)
