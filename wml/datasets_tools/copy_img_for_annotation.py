import argparse
import glob
import os.path as osp
import wml.wml_utils as wmlu
from wml.iotoolkit import get_auto_dataset_suffix
import shutil

'''
指定标注文件目录，从图像文件目标中拷贝base name相同的图像文件到标注文件目录
'''
def parse_args():
    parser = argparse.ArgumentParser(
        description='arguments')
    parser.add_argument('ann_dir', default=None,type=str,help='ann_dir')
    parser.add_argument('img_dir', default=None,type=str,help='img_dir')
    parser.add_argument('--type', default="auto",type=str,help='img_dir')
    args = parser.parse_args()
    return args

def get_all_imgs(img_dir,img_suffix=".jpg;;.jpeg;;.png;;.bmp"):
    files = wmlu.get_files(img_dir,suffix=img_suffix)
    res = {}
    for f in files:
        basename = wmlu.base_name(f)
        basename = basename.replace("\\","")
        basename = wmlu.remove_non_ascii(basename)
        if basename in res:
            d = res[basename]
            if isinstance(d,str):
                if wmlu.file_md5(d) == wmlu.file_md5(f):
                    print(f"{d} and {f} is the same file.")
                    continue
                d = [d]
            res[basename] = d+[f]
        else:
            res[basename] = f
    return res

def copy_imgfiles(ann_dir,img_dir,img_suffix=".jpg",ann_type=".xml"):
    if ann_type == "auto":
        ann_type = "."+get_auto_dataset_suffix(ann_dir)
    xml_files = wmlu.get_files(ann_dir,suffix=ann_type)
    all_img_files = get_all_imgs(img_dir)
    copy_nr = 0
    error_nr = 0
    not_found_nr = 0
    for xf in xml_files:
        base_name = wmlu.base_name(xf)
        base_name = wmlu.remove_non_ascii(base_name)
        print(base_name)
        if base_name in all_img_files:
            files = all_img_files[base_name]
            if not isinstance(files,list):
                cur_dir = osp.dirname(xf)
                print(f"{files} --> {cur_dir}")
                save_path = osp.join(cur_dir,wmlu.base_name(xf)+osp.splitext(files)[-1])
                shutil.copy(files,save_path)
                copy_nr += 1
            else:
                print(f"ERROR: Find multi img files for {xf}, img files {files}")
                error_nr += 1
        else:
            print(f"ERROR: Find img file for {xf} faild.")
            not_found_nr += 1

    print(f"total copy {copy_nr} files, {error_nr} multi files, {not_found_nr} not found files.")

if __name__ == "__main__":
    args = parse_args()
    copy_imgfiles(args.ann_dir,args.img_dir,ann_type=args.type)
