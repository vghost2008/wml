import wml_utils as wmlu
import argparse
import glob
import os.path as osp
import os
import shutil
from iotoolkit.common import find_imgs_for_ann_file

def parse_args():
    parser = argparse.ArgumentParser(description="build gif")
    parser.add_argument("src_dir",type=str,help="src dir")
    parser.add_argument("--ext",type=str,default=".xml",help="annotation ext")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    ann_files = wmlu.get_files(args.src_dir,args.ext)
    files2remove = []
    for ann_file in ann_files:
        img_file = find_imgs_for_ann_file(ann_file)
        if img_file is None:
            files2remove.append(ann_file)
    print(f"Files to remove {len(files2remove)}:")
    wmlu.show_list(files2remove)
    x = input("Y/n\n")
    if x.lower()=="y":
        print(f"Remove files")
        for file in files2remove:
            os.remove(file)
    else:
        print(f"Cancel")


    


    
