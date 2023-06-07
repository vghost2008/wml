import wml_utils as wmlu
import argparse
import glob
import os.path as osp
import os
import shutil
import img_utils as wmli

def parse_args():
    parser = argparse.ArgumentParser(description="build gif")
    parser.add_argument("src_dir",type=str,help="src dir")
    parser.add_argument("--ext",type=str,default="xml",help="annotation ext")
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
    imgs = wmlu.get_files(args.src_dir,wmli.BASE_IMG_SUFFIX)
    files2remove = []
    for img in imgs:
        ann_file = wmlu.change_suffix(img,args.ext)
        if not osp.exists(ann_file):
            files2remove.append(img)
    print(f"Files to remove {len(files2remove)}:")
    wmlu.show_list(files2remove)
    x = input("Y/n\n")
    if x.lower()=="y":
        print(f"Remove files")
        for file in files2remove:
            os.remove(file)
    else:
        print(f"Cancel")


    


    
