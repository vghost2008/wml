import wml_utils as wmlu
import argparse
import glob
import os.path as osp
import os
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description="build gif")
    parser.add_argument("src_dir",type=str,help="src dir")
    parser.add_argument("--ext",type=str,default=".xml",help="img ext")
    args = parser.parse_args()
    return args

def find_imgs_for_ann_file(ann_path):
    ann_path = osp.abspath(ann_path)
    img_suffix = [".jpg",".jpeg",".bmp",".png",".gif"]
    pattern = wmlu.change_suffix(ann_path,"*")
    files = glob.glob(pattern)
    img_file = None
    for file in files:
        if file==ann_path:
            continue
        if osp.splitext(file)[1].lower() in img_suffix:
            img_file = file
        else:
            print(f"WARNING: Unknow img format file {file} for {ann_path}")
            img_file = file
    return img_file

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


    


    