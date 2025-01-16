import wml.wml_utils as wmlu
import argparse
import os.path as osp
import os
import shutil
import wml.img_utils as wmli
from wml.iotoolkit.common import find_imgs_for_ann_file

'''
删除没有标注的图像或者没有图像的标注
'''

def parse_args():
    parser = argparse.ArgumentParser(description="build gif")
    parser.add_argument("src_dir",type=str,help="src dir")
    parser.add_argument("--ext",type=str,default="xml",help="annotation ext")
    parser.add_argument("--save-dir",type=str,help="save dir")
    parser.add_argument('--type', type=int,default=0,
                                  choices=['0', '1', '2'],help="0: remove imgs without annotations or annotation files without img; \
                                                                1: remove imgs without annotation;\
                                                                2: remove annotations without img")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    op_type = args.type
    imgs = wmlu.get_files(args.src_dir,wmli.BASE_IMG_SUFFIX)
    ann_files = wmlu.get_files(args.src_dir,args.ext)

    files2remove = []
    if op_type in [0,1]:
        #删除没有标注的图像
        for img in imgs:
            ann_file = wmlu.change_suffix(img,args.ext)
            if not osp.exists(ann_file):
                files2remove.append(img)

    if op_type in [0,2]:
        #删除没有图像的标注
        for ann_file in ann_files:
            img_file = find_imgs_for_ann_file(ann_file)
            if img_file is None:
                files2remove.append(ann_file)
    
    files2remove.sort()
    all_files_nr = len(imgs)+len(ann_files)

    save_dir = args.save_dir
    if save_dir is None:
        print(f"Files to remove {len(files2remove)}:")
        wmlu.show_list(files2remove)
        x = input(f"Y/n to remove {len(files2remove)}/{all_files_nr}\n")
    else:
        print(f"Files to move {len(files2remove)}:")
        wmlu.show_list(files2remove)
        x = input(f"Y/n to move {len(files2remove)}/{all_files_nr} to {save_dir}\n")
    if x.lower()=="y":
        if save_dir is None:
            print(f"Remove files")
            for file in files2remove:
                os.remove(file)
        else:
            os.makedirs(save_dir,exist_ok=True)
            for file in files2remove:
                bn = osp.basename(file)
                dst_path = osp.join(save_dir,bn)
                shutil.move(file,dst_path)
    else:
        print(f"Cancel")


    


    
