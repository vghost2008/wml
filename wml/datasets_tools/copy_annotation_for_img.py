import argparse
import glob
import os.path as osp
import wml.wml_utils as wmlu
from wml.iotoolkit import get_auto_dataset_suffix
import wml.img_utils as wmli
import shutil

'''
指定标注文件目录，从图像文件目标中拷贝base name相同的图像文件到标注文件目录
'''
def parse_args():
    parser = argparse.ArgumentParser(
        description='arguments')
    parser.add_argument('img_dir', default=None,type=str,help='img_dir')
    parser.add_argument('ann_dir', default=None,type=str,help='ann_dir')
    parser.add_argument('--type', default="auto",type=str,help='img_dir')
    parser.add_argument('-l','--level', default=0,type=int,help='test parent level number')
    args = parser.parse_args()
    return args

def get_name_key(path,level=0):
    path = osp.abspath(path)
    bn = wmlu.base_name(path)
    if 0 == level:
        return bn
    names = path.split(osp.sep)[:-1]
    names = names[-level:]
    names = names+[bn]
    dir_name = str(osp.sep).join(names)
    return dir_name

def get_all_ann(img_dir,level=0,suffix=".json;;.xml"):
    files = wmlu.get_files(img_dir,suffix=suffix)
    res = {}
    for f in files:
        bn = get_name_key(f,level)
        if bn in res:
            ov = res[bn]
            if not isinstance(ov,(list,tuple)):
                ov = [ov]
            res[bn] = ov+[f]
        else:
            res[bn] = f
    return res

def copy_annfiles(ann_dir,img_dir,level=0,img_suffix=".jpg",ann_type=".xml"):
    img_files = wmlu.get_files(img_dir,suffix=wmli.BASE_IMG_SUFFIX)
    all_ann_files= get_all_ann(ann_dir,level=level,suffix=ann_type)
    copy_nr = 0
    error_nr = 0
    not_found_nr = 0
    for f in img_files:
        base_name = get_name_key(f,level)
        if base_name in all_ann_files:
            files = all_ann_files[base_name]
            if not isinstance(files,list):
                target = wmlu.change_suffix(f,ann_type)
                print(f"{files} -> {target}")
                shutil.copy(files,target)
                copy_nr += 1
            else:
                print(f"ERROR: Find multi img files for {f}, ann files {files}")
                error_nr += 1
        else:
            print(f"ERROR: Find img file for {f} faild.")
            not_found_nr += 1

    print(f"total copy {copy_nr} files, {error_nr} multi files, {not_found_nr} not found files.")

if __name__ == "__main__":
    args = parse_args()
    if args.type == "auto":
        args.type = get_auto_dataset_suffix(args.ann_dir)
    copy_annfiles(args.ann_dir,args.img_dir,level=args.level,ann_type=args.type)
